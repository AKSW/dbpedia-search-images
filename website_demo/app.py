
import os
from flask import Flask, render_template, request

from kg_query.main import RankedList

app = Flask(__name__, static_folder='/scratch/sid/dataset')

app.config['UPLOAD_FOLDER'] = '.'
ranked_list = RankedList()


@app.route('/')
def upload_file_():
    return render_template('upload.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        print(f'Path: {path}!')
        images_ranked_list, uris_ranked_list = ranked_list.generate_img_results(
                path
            )
        images_path = [
            "/".join(item.split('/')[-2:]) for item in images_ranked_list
        ]
        return render_template(
            'results.html',
            image_list=images_path,
            uris_list=uris_ranked_list,
            iteration = len(images_path),
        )

if __name__ == '__main__':
   app.run(debug = True)
