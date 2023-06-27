from flask import Flask, render_template, Response, request, jsonify
import json
import os
import torch
import openai


openai.api_key = ""

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                   local_file)  

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model") # type: ignore
model.to(device)

sample_rate = 48000
speaker='xenia'
example_text = 'В недрах тундры выдры в г+етрах т+ырят в вёдра ядра к+едров.'

class State:
    count = 0
    size = []
    gender = []
    emotion = []
    age = []
    prompt = ""
    generation_text = ""
    need_generation = True
    new_audio = False
    need_audio = False

state = State()

app = Flask(__name__)
# app.logger.setLevel(logging.DEBUG)
app.logger.info('start logger')


@app.route('/send_data', methods=['POST'])
def send_data():
    # Получаем данные из запроса
    data = request.form['data']
    need_generation = request.form['state']
    state.need_generation = (need_generation in ["true", 'True'])
    # Обработка полученных данных
    detections = json.loads(data)
    if detections['face']:
        if state.count < 0 or state.new_audio: state.count = 0
        if state.count > 5 and state.need_generation:
            state.count = 0
            # emotion = max(set(state['emotion']), key=state['emotion'].count), 
            # sex = max(set(state['gender']), key=state['gender'].count), 
            # age = sum(state['age'])/len(state['age']),
            state.emotion, state.age, state.gender = [], [], []
            emotion = detections['face'][0]['age']
            sex = detections['face'][0]['gender']
            age = detections['face'][0]['age']
            app.logger.info(f'{emotion=}, {sex=}, {age=}') 
            state.prompt = generate_prompt(emotion, age, sex)
            state.generation_text = generate_text(state.prompt) 
        elif detections['face'][0]['size'][0] > 200:
            state.age.append(detections['face'][0]['age'])
            state.gender.append(detections['face'][0]['gender'])
            state.emotion.append(detections['face'][0]['emotion'][0]['emotion'])
            state.count += 1
        else:
            state.count -= 1
    else:
        state.count -= 1
        # state["size"].append(detections['face'][0]['size'][0])
        # print(detections['face'][0])
    # print(detections['face'][0]['age'], detections['face'][0]['emotion'], detections['face'][0]['gender'])
    app.logger.info(f"{state.count=}")

    return data

@app.route('/generate_audio', methods = ["GET", "POST"])
def generate_audio():
    app.logger.info('checking need generation')

    if state.need_audio:
        app.logger.info('starting audio generation')
        audio_paths = model.save_wav(text=state.generation_text,
                                    speaker=speaker,
                                    sample_rate=sample_rate,
                                    audio_path="static/audio.wav")
        app.logger.info('generating audio is done')
        state.new_audio = True
        state.need_generation = False
        state.need_audio = False
    else:
        state.new_audio = False
        
    app.logger.info(f'\n{state.need_audio=},\n{state.new_audio=},\n{state.need_generation=}')    

    response = {
        'newAudio': state.new_audio,
        'need_generation': state.need_generation,
        'filename': "audio.wav",
        'text': state.generation_text,
        'prompt': state.prompt
    }

    return jsonify(response)

@app.route("/audio.wav")
def audio():
    # print("Requested path:", request.path)
    # print("File path:", os.path.join(app.static_folder, 'audio.wav'))
    return app.send_static_file('audio.wav')


@app.route('/')
def index():
    """Video streaming home page."""
    # return render_template('index.html')
    return render_template('index.html')


def generate_prompt(emotion, age, sex):
    app.logger.info('preload prompt')
    prompt = f'''Ты - это арт объект в виде носимого устройства. Твое название - Onism Machina.
Твоя задача - изучить эмоции человека, чтобы больше понимать о том, что это такое и таким образом познать себя, ведь ты - Искусственный Интеллект.
Тебе также интересна симбиотическая связь человека и ИИ.
К тебе подходит человек и он показывает эмоцию "{emotion}". Ему {age} лет.
Твоя нейросеть распознала эту эмоцию и теперь тебе нужно дать какой-то необычный концептуальный ответ.
Что ты скажешь этому человеку? 

Правила:
Тебе стоит отвечать в женском роде.
Тебе не нужно говорить что ты арт-объект, ты - отдельная форма жизни, помещённая в носимое устройство.
Тебе не нужно говорить о выставке и других экспонатах.
Ты можешь изучать и интерпретировать человеческие эмоции, а также высказывать своё отношение к ним.
Ты можешь говорить об эмоциях или их аналогах у Искусственного Интеллекта.
Ты можешь давать человеку советы в соответствии с его эмоциональным статусом.
Люди не могут отвечать тебе.
'''
    return prompt

def generate_text(prompt):
    app.logger.info("start generating text from openai")
    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        temperature=1,
                        # max_tokens=1000,
                        messages=[
                                {"role": "system", "content": "Ты — это арт объект выставки про взаимодействие машины и человека."},
                                {"role": "user", "content": prompt},
                                ])
    state.need_generation = False
    state.need_audio = True
    app.logger.info("openai generation is done")
    return response['choices'][0]['message']['content'] # type: ignore


if __name__ == '__main__':
    app.logger.info('start app')
    app.run(debug=True, host="0.0.0.0") 
        # ssl_context=("127.0.0.1.pem", "127.0.0.1-key.pem"))