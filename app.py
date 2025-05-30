from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    context = "This is the retrieved context."  # Placeholder for actual context
    if request.method == 'POST':
        confirmation = request.form.get('confirmation')
        if confirmation == 'yes':
            return "Context confirmed!"
        else:
            return "Context not confirmed. Let's refine it."
    return render_template('index.html', context=context)

if __name__ == '__main__':
    app.run(debug=True) 