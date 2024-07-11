from flask import Flask, request, render_template, redirect, url_for, jsonify, stream_with_context, Response
from collections import Counter, defaultdict
import scholar_analyzer
import json
import hashlib

app = Flask(__name__)
cache = {}
log_messages = []

@app.route('/', methods=['GET', 'POST'])
def index():
    
    log_messages.clear()
    search_performed = False
    if request.method == 'POST':
        name = request.form['author_name']
        start_year = request.form['start_year']
        end_year = request.form['end_year']

        # convert start_year and end_year to int
        try:
            start_year = int(start_year)
            end_year = int(end_year)
        except ValueError:
            return render_template('index.html', cache=cache, error="Please enter a valid year")

        # Search for authors
        search_performed = True
        authors_list = scholar_analyzer.resolve_author(name)
        
        # Pass the list of authors to the template
        return render_template('index.html', 
                               authors_list=authors_list, 
                               cache=cache, 
                               search_performed=search_performed,
                               startYear=start_year,
                               endYear=end_year)
    return render_template('index.html', cache=cache)

@app.route('/delete/<cache_key>', methods=['POST'])
def delete_result(cache_key):
    # Extract current page info from the form
    current_author_id = request.form.get('current_author_id')
    current_start_year = request.form.get('current_start_year')
    current_end_year = request.form.get('current_end_year')

    # Check if the item being deleted is the current page
    if cache_key in cache:
        item = cache[cache_key]
        if (str(item['authorId']) == current_author_id and
            str(item['start_year']) == current_start_year and
            str(item['end_year']) == current_end_year):
            # If deleting the current page, redirect to index
            del cache[cache_key]
            with open('cache.json', 'w') as f:
                json.dump(cache, f)
            return redirect(url_for('index'))
        else:
            # If not, delete and stay on the current page
            del cache[cache_key]
            with open('cache.json', 'w') as f:
                json.dump(cache, f)
            return redirect(request.referrer)
    return redirect(url_for('index'))

def get_results(authorId, author_name, start_year, end_year):

    start_year = int(start_year)
    end_year = int(end_year)
    result = {
        "authorId": authorId,
        "start_year": start_year,
        "end_year": end_year,
        "papers": [],
        "analysis": "",
        "author_name": author_name,
        "coauthors_histogram": {},
        "paper_topics": {},
    }

    log_messages.append("Getting author details...")
    # author_props = scholar_analyzer.author_details(authorId)
    # author_name = author_props['name']

    log_messages.append("Fetching papers...")
    papers, coauthors_histogram = scholar_analyzer.get_papers(authorId, author_name, start_year, end_year)
    if not papers:
        if start_year == end_year:
            result["error"] = f"No papers found for {author_name} in {start_year}"
        else:
            result["error"] = f"No papers found for {author_name} between {start_year} and {end_year}"
        return result
    result["coauthors_histogram"] = coauthors_histogram

    log_messages.append("Analyzing research...")
    abstracts = [f"{paper['title']}\n{paper['abstract']}" for paper in papers if paper['abstract']]
    abstracts_string = "\n\n".join(abstracts)
    result["analysis"] = scholar_analyzer.analyze_topics(abstracts_string)

    paper_years = [int(paper['year']) for paper in papers if paper['abstract']]

    log_messages.append("Discovering topics...")
    topics, embs = scholar_analyzer.get_topics(abstracts)

    log_messages.append("Discovering trends...")
    clusters = scholar_analyzer.cluster_topics(topics, embs)
    final_papers = []
    i_cnt = 0
    for paper in papers:
        if paper['abstract']:
            paper["topic"] = clusters[i_cnt]
            i_cnt += 1
        else:
            paper["topic"] = "Unknown"
        final_papers.append(paper)
    result["papers"] = final_papers
    cluster_trends = defaultdict(Counter)
    for i, cluster in enumerate(clusters):
        cluster_trends[cluster][paper_years[i]] += 1
    for cluster in cluster_trends:
        for year in range(start_year, end_year + 1):
            if year not in cluster_trends[cluster]:
                cluster_trends[cluster][year] = 0
    cluster_trends = dict(cluster_trends)
    result["cluster_trends"] = cluster_trends
    return result

@app.route('/stream-logs')
def stream_logs():
    def generate():
        while True:
            if log_messages:
                msg = log_messages.pop(0)  # Get the first message in the list
                yield f"data: {msg}\n\n"
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/results')
def results():
    authorId = request.args.get('author_id')
    author_name = request.args.get('author_name')
    start_year = request.args.get('start_year')
    end_year = request.args.get('end_year')
    index = request.args.get('index', -1)
    
    cache_key = hashlib.md5(f"{authorId}{start_year}{end_year}".encode()).hexdigest()
    
    if cache_key in cache:
        result = cache[cache_key]
    else:
        try:
            result = get_results(authorId, author_name, start_year, end_year)
        except Exception as e:
            raise e
            return render_template('error.html', error=e)
        result['index'] = index
        cache[cache_key] = result

    # update cache on disk
    with open('cache.json', 'w') as f:
        json.dump(cache, f)
    try:
        return render_template('results.html', result=result, cache=cache, pretty_name=author_name, index=index)
    except Exception as e:
        raise e
        return render_template('error.html', error=e)

@app.route('/logs')
def logs():
    # Return the log messages as JSON
    return jsonify(log_messages)

if __name__ == '__main__':
    # if cache on disk, load it
    try:
        with open('cache.json', 'r') as f:
            cache = json.load(f)
    except FileNotFoundError:
        pass
    app.run(debug=True)