import os
from flask import Flask, flash, request, redirect, url_for, Response
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = {'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4'))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Subtitle Generator</title>
    <h1>Upload Video File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    <h1>Export Subtitles</h1>
    <a href="/export">Export SRT</a>
    '''
    
@app.route("/export")
def export():
    import os
    import torch
    import zipfile
    import torchaudio
    from glob import glob

    device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
    model, decoder, utils = torch.hub.load('snakers4/silero-models',
                                        model='silero_stt',
                                        language='en')
    (read_batch, split_into_batches,
    read_audio, prepare_model_input) = utils  # see function signature for details
    
    
    os.system("ffmpeg -i 'video.mp4' -vn -acodec copy audio.aac")
    os.system("ffmpeg -i audio.aac audio.wav")


    # download a single file, any format compatible with TorchAudio (soundfile backend)
    # torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
    #                                dst ='speech_orig.wav', progress=True)
    test_files = glob('audio.wav') 
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]))

    text = ""
    output = model(input)
    for example in output:
        pred = decoder(example.cpu())
        text = text + pred
        
    os.system("curl -LJO https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt")
    os.system("curl -LJO https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_bigramdictionary_en_243_342.txt")



    import pkg_resources
    from symspellpy import SymSpell, Verbosity

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    # term_index is the column of the term and count_index is the
    # column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    # lookup suggestions for multi-word input strings (supports compound
    # splitting & merging)
    # input_term = ("whereis th elove hehad dated forImuch of thepast who "
    #              "couqdn'tread in sixtgrade and ins pired him")
    # max edit distance per lookup (per single word, not per whole input string)
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    # display suggestion term, edit distance, and term frequency
    for suggestion in suggestions:
        print(suggestion)
        
        
    text = str(suggestion)


    cnt = 0
    textlines = []
    while cnt < len(text.split(" ")):
        print(text.split(" ")[cnt:cnt+5])
        line = "\n" + " ".join(text.split(" ")[cnt:cnt+5])
        textlines.append(line)
        cnt += 5
        
        
    f = open("script_cleaned.txt", "a")
    f.writelines(textlines)
    f.close()


    os.system("python -m aeneas.tools.execute_task \
        audio.wav \
        script_cleaned.txt \
        'task_language=eng|os_task_file_format=srt|is_text_type=plain' \
        subtitles.srt")



    with open("subtitles.srt") as f:
        srt = f.read()
        
    return Response(
        srt,
        mimetype="text/srt",
        headers={
            "Content-disposition": "attachment; filename=subtitiles.srt"
        }
    )
		
if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug = True)
