import flask

import vsearch

app = flask.Flask(__name__, template_folder='./templates')


@app.route('/')
def hello() -> str:
    return 'I just want to fuck your juicy pussy!'


@app.route('/search4', methods=['POST', 'GET'])
def do_search():
    phrase = flask.request.form['phrase']
    letters = flask.request.form['letters']
    title = 'Here are your results'
    results = str(vsearch.search4letters(phrase, letters))
    if len(results):
        results = 'Nothing found'
        return flask.render_template('result.html', the_phrase=phrase, the_letters=letters, the_title=title,
                                     the_results=results)


@app.route('/entry')
def entry_page():
    return flask.render_template('entry.html', the_title='Welcome to search4letters on web')


if __name__ == '__main__':
    app.run(debug=True)

'''当程序被导入其他地方应用的时候，__name__为该程序的路径，如果直接运行，则为'__main__'''
