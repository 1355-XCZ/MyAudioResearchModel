import soundfile as sf
import os

def slice(auido_path,s,e):
    audio,sr = sf.read(auido_path)
    audio = audio[int(s*sr):int(e*sr)]
    sf.write(auido_path,audio,sr)

silce_dict = {
    "55":{
        "s":0.5,
        "e":10.5
    },
    "59":{
        "s":10,
        "e":20
    },
    "62":{
        "s":0,
        "e":12
    },
    "63":{
        "s":2,
        "e":12
    },
    "64":{
        "s":4,
        "e":14
    }
    
}
audio_dir = "/home/node50_ssd/zkliu/git/FlowSE/static/audio"

for key in silce_dict.keys():
    slice(f"{audio_dir}/noisy/{key}.wav",silce_dict[key]['s'],silce_dict[key]['e'])
    slice(f"{audio_dir}/clean/{key}.wav",silce_dict[key]['s'],silce_dict[key]['e'])
    slice(f"{audio_dir}/enhanced/{key}.wav",silce_dict[key]['s'],silce_dict[key]['e'])
    slice(f"{audio_dir}/enhanced_wotext/{key}.wav",silce_dict[key]['s'],silce_dict[key]['e'])