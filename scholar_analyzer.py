import requests
from datetime import datetime
import math
from sklearn.cluster import KMeans
import openai
import os
from openai import OpenAI
from collections import Counter, defaultdict
from thefuzz import fuzz
import logging

logger = logging.getLogger(__name__)


openai.api_key = os.environ['OPENAI_API_KEY']
ss_headers = {'x-api-key': os.environ['SEMANTICSCHOLAR_API_KEY']}

gptclient = OpenAI()

def gpt_embedding(text):
    response = gptclient.embeddings.create(
    input=text,
    model="text-embedding-3-large")
    return response.data[0].embedding

def gpt_call(prompt_text, model="gpt-4o", max_tokens=256, model_client=None):
        model_response = gptclient.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0
            )
        return model_response.choices[0].message.content

def author_fields_of_study(authorId):
    rsp = requests.get(f'https://api.semanticscholar.org/graph/v1/author/{authorId}', params={'fields': 'papers.s2FieldsOfStudy'}, headers=ss_headers)
    rsp.raise_for_status()
    fields_counter = []
    for paper in rsp.json()['papers']:
        for field in paper['s2FieldsOfStudy']:
            fields_counter.append(field['category'])
    fields_counter = Counter(fields_counter).most_common(2)
    return [field[0] for field in fields_counter]

def resolve_author(name):
    rsp = requests.get('https://api.semanticscholar.org/graph/v1/author/search',
                        params={'query': f"{name}", 'fields': 'authorId,name,affiliations,citationCount'},
                        headers=ss_headers)
    rsp.raise_for_status()

    results = rsp.json()
    if results['total'] == 0:  # no results
        return []
    # get fuzzy match score for each author
    # for author in results['data']:
    #     author['fuzzy_score'] = fuzz.ratio(name.lower(), author['name'].lower())
    # # sort by fuzzy score and citation count
    # results['data'] = sorted(results['data'], key=lambda x: (x['fuzzy_score'], x['citationCount']), reverse=True)[:5]
    results['data'] = sorted(results['data'], key=lambda x: x['citationCount'], reverse=True)[:5]
    
    if len(results['data']) <= 1:
        return results['data']
    
    results_final = []
    for author in results['data']:
        author['fields'] = author_fields_of_study(author['authorId'])
        results_final.append(author)
    return results_final

def author_details(author_id):
    rsp = requests.get(f'https://api.semanticscholar.org/graph/v1/author/{author_id}',
                       params={'fields': 'authorId,name,affiliations,homepage'},
                       headers=ss_headers)
    rsp.raise_for_status()
    return rsp.json()

def get_author_papers(authorId, limit=250):
    rsp = requests.get(f'https://api.semanticscholar.org/graph/v1/author/{authorId}/papers',
                       params={'fields': 'year', 'limit': 250},
                       headers=ss_headers)
    rsp.raise_for_status()
    return rsp.json()['data']

def get_paper_details(paperIds):
    rsp = requests.post('https://api.semanticscholar.org/graph/v1/paper/batch',
                       params={'fields': 'abstract,title,url,year,authors'},
                       json={'ids': paperIds},
                       headers=ss_headers)
    rsp.raise_for_status()
    return rsp.json()

def get_papers(authorId, authorName, start_year, end_year):
    papers = get_author_papers(authorId)
    coauthor_counter = Counter()
    papers = [paper for paper in papers if paper['year']]
    papers = [paper for paper in papers if start_year <= int(paper['year']) <= end_year]
    if not papers:
        return [], {}
    paperIds = [paper['paperId'] for paper in papers]
    paper_details = get_paper_details(paperIds)
    for paper in paper_details:
        coauthor_counter.update([author['name'] for author in paper['authors'] if author['name'] != authorName])
    coauthor_counter = dict(coauthor_counter)
    papers_details_minimal = []
    for paper in paper_details:
        authors = ", ".join([author['name'] for author in paper['authors']])
        papers_details_minimal.append({
            "title": paper['title'],
            "authors": authors,
            "year": paper['year'],
            "url": paper['url'],
            "abstract": paper['abstract']
        })
    return papers_details_minimal, coauthor_counter

def per_abstract_topics(abstract):
    prompt = f"You are a helpful assistant, helping a prospective PhD student learn about the research areas of different researchers. You are given the abstract of a paper. Analyze the provided abstract and identify the main research areas. Provide a single sentence description of the research topic of the abstract. The abstract is:\n{abstract}\n\nTopic description:"
    response = gpt_call(prompt)

    # get embedding of the response
    embedding = gpt_embedding(response)
    return response, embedding

def get_cluster_topic(cluster_topics):
    prompt = f"You are a helpful assistant, helping a prospective PhD student learn about the research areas of different researchers. You are given a list of research topics. Return a two or three word title for the list which should reflect the topics as best as possible. \nResearch areas:\n{cluster_topics}"
    response = gpt_call(prompt)
    return response

def get_topics(abstracts):
    topics = []
    embeddings = []
    for abstract in abstracts:
        topic, embedding = per_abstract_topics(abstract)
        topics.append(topic)
        embeddings.append(embedding)
    return topics, embeddings

def cluster_topics(topics, embeddings):
    # automatically choose k for kmeans
    min_topics = min(3, len(topics))
    max_topics = int(math.ceil(math.sqrt(len(topics))))
    for k in range(min_topics, max(max_topics, min_topics+1)):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        # elbow method
        if kmeans.inertia_ < 0.1:
            break

    clusters = {}
    for i, label in enumerate(kmeans.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(topics[i])

    cluster_topics = {}
    for label, cluster in clusters.items():
        cluster_topics[label] = get_cluster_topic(cluster)

    # invert the cluster mapping
    inverted_clusters = []
    for label in kmeans.labels_:
        inverted_clusters.append(cluster_topics[label])

    return inverted_clusters

def analyze_topics(abstracts):
    prompt = f"""You are a helpful assistant, helping a prospective PhD student learn about the research areas of different researchers. You are given a list of abstracts of a researcher's papers. 
    Analyze the provided abstracts and identify the main research areas. 
    Provide a list of short sentences, each describing a research topic that appears in the abstracts. 
    Do not provide too general topics, or topics that use fluff words or buzz words. 
    
    Examples of bad topics which should not be given: 
    'Advancements in X', 'Innovations in Y', 'Advanced Z'. 

    Examples of good topics which should be given: 
    'Generalization under distribution shifts', 'Evaluation of multimodal translation models', 'Echolocation of fruit flies', 'Effects of shock therapy on the human nervous system'. 
    
    The abstracts are:
    {abstracts}
    
    Research areas:
    """
    response = gpt_call(prompt)
    outlier_prompt = f"You are a helpful assistant, helping a prospective PhD student learn about the research areas of different researchers. You are given a list of research topics. Some of the topic may be outliers compared to others. Return a list of the topics which are not outliers, but are instead related to the majority. Do NOT return any of the irrelevant topics. Do not blabber or reply with any explanation. Renumber the list so it starts at 1 and increments for each list item. \nResearch areas:\n{response}"
    response = gpt_call(outlier_prompt)
    return response

# def main(authorId, start_year, end_year):
#     start_year = int(start_year)
#     end_year = int(end_year)
#     result = get_result(authorId, start_year, end_year)
#     return result
    

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Analyze a researcher's recent papers")
#     parser.add_argument("first_name", help="Researcher's first name")
#     parser.add_argument("last_name", help="Researcher's last name")
#     parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
#     parser.add_argument("end_date", help="End date (YYYY-MM-DD)")
#     args = parser.parse_args()

#     start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
#     end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

#     result = main(args.first_name, args.last_name, start_date, end_date)
#     print(result)
