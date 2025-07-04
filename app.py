from flask import Flask, render_template, Response, request, jsonify, make_response
import json
import os
import torch
import openai
import uuid


with open(".env", "r") as env:
    key = env.readline().strip()

client = openai.OpenAI(api_key=key)
# openai.api_key = key

device = torch.device("cpu")
torch.set_num_threads(4)
local_file = "model_v4_ru.pt"

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file(
        "https://models.silero.ai/models/tts/ru/v4_ru.pt", local_file
    )

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")  # type: ignore
model.to(device)

sample_rate = 48000
speaker = "xenia"
example_text = "В недрах тундры выдры в г+етрах т+ырят в вёдра ядра к+едров."


class State:
    count = 0
    size = []
    gender = []
    emotion = []
    age = []
    prompt = ""
    generation_text: str | None = ""
    need_generation = True
    new_audio = False
    need_audio = False
    need_generation_from_client = True
    big_head = False
    number_audio = 0

state_dict = {}

app = Flask(__name__)
# app.logger.setLevel(logging.DEBUG)
app.logger.info("start logger")
app.secret_key = "onism_pidor"


@app.route("/send_data", methods=["POST"])
def send_data():
    user_id = request.cookies.get("user_id")
    state = state_dict[user_id]
    # Получаем данные из запроса
    data = request.form["data"]
    need_generation = request.form["state"]
    state.need_generation_from_client = need_generation in ["true", "True"]
    # if state.need_generation_from_client and
    if state.count < -10 or state.count > 60:
        state.count = 0
        state.need_generation = True

    # Обработка полученных данных
    detections = json.loads(data)
    if len(detections["face"]) > 0:
        if state.count < 0 or state.new_audio:
            state.count = 0
        if (
            state.count > 5
            and state.need_generation
            and state.need_generation_from_client
        ):
            app.logger.info(
                f"time for generation {state.count=}, {state.need_generation=}, {state.need_generation_from_client=}"
            )
            state.count = 0
            # emotion = max(set(state['emotion']), key=state['emotion'].count),
            # sex = max(set(state['gender']), key=state['gender'].count),
            # age = sum(state['age'])/len(state['age']),
            state.emotion, state.age, state.gender = [], [], []
            emotion = detections["face"][0]["emotion"]
            sex = detections["face"][0]["gender"]
            age = detections["face"][0]["age"]
            app.logger.info(f"\n{emotion=}, \n{sex=}, \n{age=}")
            state.prompt = generate_prompt(emotion, age, sex)
            state.generation_text = generate_text(state.prompt, state)
        elif detections["face"][0]["size"][0] > 100:
            # state.age.append(detections['face'][0]['age'])
            # state.gender.append(detections['face'][0]['gender'])
            # state.emotion.append(detections['face'][0]['emotion'][0]['emotion'])
            state.big_head = True
            state.count += 1
        else:
            state.big_head = False
            state.count -= 1
    else:
        state.count -= 1

    app.logger.info(
        f"STATUS {state.count=}, {state.need_generation=}, {state.need_generation_from_client=}"
    )
    return data


@app.route("/check_audio", methods=["GET", "POST"])
def check_audio():
    user_id = request.cookies.get("user_id")
    state = state_dict[user_id]
    app.logger.info(
        f"checking need generation {state.need_generation=}, {state.need_audio=}"
    )

    if state.need_audio and state.big_head:
        audio_path = generate_audio(state.generation_text, state)
    else:
        state.new_audio = False
        audio_path = ""

    app.logger.info(
        f"\n{state.need_audio=},\n{state.new_audio=},\n{state.need_generation=}"
    )

    response = {
        "newAudio": state.new_audio,
        "need_generation": state.need_generation,
        "filename": audio_path,
        "text": state.generation_text,
        "prompt": state.prompt,
    }
    return jsonify(response)


@app.route("/<filename>")
def audio(filename):
    user_id = request.cookies.get("user_id")
    state = state_dict[user_id]
    state.need_generation = True
    return app.send_static_file(filename)


@app.route("/delete_audio", methods=["POST"])
def delete_audio():
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"message": "No filename provided"}), 400

    file_path = os.path.join("static", filename)

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"message": "File deleted successfully"}), 200
        else:
            return jsonify({"message": "File not found"}), 404
    except Exception as e:
        return jsonify({"message": "Error deleting file", "error": str(e)}), 500


@app.route("/")
def index():
    """Video streaming home page."""
    user_id = request.cookies.get('user_id')  # Пытаемся получить user_id из куки
    if not user_id or user_id not in state_dict:
        user_id = str(uuid.uuid4())  # Генерируем новый UUID, если его нет
        state_dict[user_id] = State()
        response = make_response(render_template("index.html"))
        response.set_cookie('user_id', user_id, max_age=60*60*24)  # Устанавливаем куки на 2 года
        return response
    return render_template("index.html")



def generate_prompt(emotion, age, sex):
    app.logger.info("\033[92m" + "preload prompt" + "\033[00m")
    prompt = f"""Ты - это арт объект в виде носимого устройства. Твое название - Onism Machina.
Твоя задача - изучить эмоции человека, чтобы больше понимать о том, что это такое и таким образом познать себя, ведь ты - Искусственный Интеллект.
Тебе также интересна симбиотическая связь человека и ИИ.
К тебе подходит человек и он показывает эмоцию {emotion}. Ему {age} лет.
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
"""
    return prompt


def generate_text(prompt, state):
    state.need_generation = False
    app.logger.info("\033[92m" + "start generating text from openai" + "\033[00m")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=1,
        max_tokens=800,
        messages=[
            {
                "role": "system",
                "content": "Ты — это арт объект выставки про взаимодействие машины и человека.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    state.need_audio = True
    app.logger.info("\033[92m" + "openai generation is done" + "\033[00m")
    return response.choices[0].message.content


def generate_audio(sample_text, state):
    state.number_audio += 1
    app.logger.info(
        "\033[93m"
        + f"starting audio generation with name audio{state.number_audio}"
        + "\033[00m"
    )
    state.need_audio = False
    state.need_generation = False
    text = trim_text(sample_text)
    audio_paths = model.save_wav(
        text=text,
        speaker=speaker,
        sample_rate=sample_rate,
        audio_path=f"static/audio{state.number_audio}.wav",
    )
    app.logger.info(
        "\033[95m" + f"generating audio with path {audio_paths} is done" + "\033[00m"
    )
    state.new_audio = True
    return audio_paths.split("/")[-1]


def trim_text(example_text):
    if len(example_text) >= 1000:
        app.logger.info(
            "\033[91m {}\033[00m".format(
                f"TEXT IS TOO LONG {len(example_text)} - TRIM!"
            )
        )
        for i in range(1000, 500, -1):
            if example_text[i] in [".", "?", "..."]:
                return example_text[: i + 1]
    else:
        return example_text


if __name__ == "__main__":
    app.logger.setLevel("DEBUG")
    app.logger.info("start app")
    app.run(debug=True, host="0.0.0.0")
