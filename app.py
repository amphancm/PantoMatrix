import os
import torch
import spaces
import librosa
import numpy as np
import gradio as gr
import soundfile as sf
import OpenGL.GL as gl
import torch.nn.functional as F

# os.system("Xvfb :99 -ac &")
# os.environ["DISPLAY"] = ":99"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"

from torchvision.io import write_video
from emage_utils import fast_render
from emage_utils.npz2pose import render2d
from emage_utils.motion_io import beat_format_save
from models.camn_audio import CamnAudioModel
from models.disco_audio import DiscoAudioModel
from models.emage_audio import EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_folder = "./gradio_results"
os.makedirs(save_folder, exist_ok=True)
print(device)

if not os.path.exists("./emage_evaltools/smplx_models"):
    import subprocess
    subprocess.run(["git", "clone", "https://huggingface.co/H-Liu1997/emage_evaltools"])

model_camn  = CamnAudioModel.from_pretrained("H-Liu1997/camn_audio").to(device).eval()
model_disco = DiscoAudioModel.from_pretrained("H-Liu1997/disco_audio").to(device).eval()

face_motion_vq   = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/face").to(device).eval()
upper_motion_vq  = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/upper").to(device).eval()
lower_motion_vq  = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/lower").to(device).eval()
hands_motion_vq  = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/hands").to(device).eval()
global_motion_ae = EmageVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/global").to(device).eval()

emage_vq_model = EmageVQModel(
    face_model=face_motion_vq, 
    upper_model=upper_motion_vq,
    lower_model=lower_motion_vq, 
    hands_model=hands_motion_vq,
    global_model=global_motion_ae
).to(device).eval()

model_emage = EmageAudioModel.from_pretrained("H-Liu1997/emage_audio").to(device).eval()


def inference_camn(audio_path, sr_model, pose_fps, seed_frames):
    audio_loaded, _ = librosa.load(audio_path, sr=sr_model)
    audio_t = torch.from_numpy(audio_loaded).float().unsqueeze(0).to(device)
    sid     = torch.zeros(1, 1).long().to(device)
    with torch.no_grad():
        motion_pred = model_camn(audio_t, sid, seed_frames=seed_frames)["motion_axis_angle"]
    t = motion_pred.shape[1]
    motion_pred = motion_pred.cpu().numpy().reshape(t, -1)
    npz_path = os.path.join(save_folder, "camn_output.npz")
    beat_format_save(npz_path, motion_pred, upsample=30 // pose_fps)
    return npz_path

def inference_disco(audio_path, sr_model, pose_fps, seed_frames):
    audio_loaded, _ = librosa.load(audio_path, sr=sr_model)
    audio_t = torch.from_numpy(audio_loaded).float().unsqueeze(0).to(device)
    sid = torch.zeros(1, 1).long().to(device)
    with torch.no_grad():
        motion_pred = model_disco(audio_t, sid, seed_frames=seed_frames, seed_motion=None)["motion_axis_angle"]
    t = motion_pred.shape[1]
    motion_pred = motion_pred.cpu().numpy().reshape(t, -1)
    npz_path = os.path.join(save_folder, "disco_output.npz")
    beat_format_save(npz_path, motion_pred, upsample=30 // pose_fps)
    return npz_path

def inference_emage(audio_path, sr_model, pose_fps):
    audio_loaded, _ = librosa.load(audio_path, sr=sr_model)
    audio_t = torch.from_numpy(audio_loaded).float().unsqueeze(0).to(device)
    sid = torch.zeros(1, 1).long().to(device)
    with torch.no_grad():
        latent_dict  = model_emage.inference(audio_t, sid, emage_vq_model, masked_motion=None, mask=None)
        face_latent  = latent_dict["rec_face"] if model_emage.cfg.lf > 0 and model_emage.cfg.cf == 0 else None
        upper_latent = latent_dict["rec_upper"] if model_emage.cfg.lu > 0 and model_emage.cfg.cu == 0 else None
        hands_latent = latent_dict["rec_hands"] if model_emage.cfg.lh > 0 and model_emage.cfg.ch == 0 else None
        lower_latent = latent_dict["rec_lower"] if model_emage.cfg.ll > 0 and model_emage.cfg.cl == 0 else None

        face_index  = torch.max(F.log_softmax(latent_dict["cls_face"], dim=2), dim=2)[1] if model_emage.cfg.cf > 0 else None
        upper_index = torch.max(F.log_softmax(latent_dict["cls_upper"], dim=2), dim=2)[1] if model_emage.cfg.cu > 0 else None
        hands_index = torch.max(F.log_softmax(latent_dict["cls_hands"], dim=2), dim=2)[1] if model_emage.cfg.ch > 0 else None
        lower_index = torch.max(F.log_softmax(latent_dict["cls_lower"], dim=2), dim=2)[1] if model_emage.cfg.cl > 0 else None

        ref_trans = torch.zeros(1, 1, 3).to(device)
        all_pred = emage_vq_model.decode(
            face_latent=face_latent, 
            upper_latent=upper_latent, 
            lower_latent=lower_latent, 
            hands_latent=hands_latent,
            face_index=face_index, 
            upper_index=upper_index, 
            lower_index=lower_index, 
            hands_index=hands_index,
            get_global_motion=True, 
            ref_trans=ref_trans[:, 0]
        )

    motion_pred = all_pred["motion_axis_angle"]
    t = motion_pred.shape[1]
    motion_pred = motion_pred.cpu().numpy().reshape(t, -1)
    face_pred   = all_pred["expression"].cpu().numpy().reshape(t, -1)
    trans_pred  = all_pred["trans"].cpu().numpy().reshape(t, -1)
    npz_path = os.path.join(save_folder, "emage_output.npz")
    beat_format_save(npz_path, motion_pred, upsample=30 // pose_fps, expressions=face_pred, trans=trans_pred)
    return npz_path


def inference_app(audio, model_type, render_mesh=False, render_face=False, render_mesh_face=False):
    if audio is None:
        return [None, None, None, None, None]

    sr_in, audio_data = audio
    # --- TRUNCATE to 60 seconds if longer ---
    max_len = int(60 * sr_in)
    if len(audio_data) > max_len:
        audio_data = audio_data[:max_len]
    # ----------------------------------------

    tmp_audio_path = os.path.join(save_folder, "tmp_input.wav")
    sf.write(tmp_audio_path, audio_data, sr_in)

    if model_type == "CaMN (Upper only)":
        sr_model, pose_fps, seed_frames = model_camn.cfg.audio_sr, model_camn.cfg.pose_fps, model_camn.cfg.seed_frames
        npz_path = inference_camn(tmp_audio_path, sr_model, pose_fps, seed_frames)
    elif model_type == "DisCo (Upper only)":
        sr_model, pose_fps, seed_frames = model_disco.cfg.audio_sr, model_disco.cfg.pose_fps, model_disco.cfg.seed_frames
        npz_path = inference_disco(tmp_audio_path, sr_model, pose_fps, seed_frames)
    else:
        sr_model, pose_fps = model_emage.cfg.audio_sr, model_emage.cfg.pose_fps
        npz_path = inference_emage(tmp_audio_path, sr_model, pose_fps)

    motion_dict = np.load(npz_path, allow_pickle=True)
    v2d_body    = render2d(motion_dict, (720, 480), face_only=False, remove_global=True)
    out_2d_body = npz_path.replace(".npz", "_2dbody.mp4")
    write_video(out_2d_body, v2d_body.permute(0, 2, 3, 1), fps=30)
    final_2d_body = out_2d_body.replace(".mp4", "_audio.mp4")
    fast_render.add_audio_to_video(out_2d_body, tmp_audio_path, final_2d_body)

    final_mesh_video     = None
    final_meshface_video = None
    if render_mesh:
        mesh_vid = fast_render.render_one_sequence_no_gt(
            npz_path, save_folder, tmp_audio_path, "./emage_evaltools/smplx_models/"
        )
        final_mesh_video = mesh_vid

    if render_mesh_face and render_mesh:
        meshface_vid = fast_render.render_one_sequence_face_only(
            npz_path, save_folder, tmp_audio_path, "./emage_evaltools/smplx_models/"
        )
        final_meshface_video = meshface_vid

    final_face_video = None
    if render_face:
        v2d_face    = render2d(motion_dict, (720, 480), face_only=True, remove_global=True)
        out_2d_face = npz_path.replace(".npz", "_2dface.mp4")
        write_video(out_2d_face, v2d_face.permute(0, 2, 3, 1), fps=30)
        final_face_video = out_2d_face.replace(".mp4", "_audio.mp4")
        fast_render.add_audio_to_video(out_2d_face, tmp_audio_path, final_face_video)

    return [final_2d_body, final_mesh_video, final_face_video, final_meshface_video, npz_path]

examples_data = [
    ["./examples/audio/2_scott_0_103_103_10s.wav", "DisCo (Upper only)", True, True, True],
    ["./examples/audio/2_scott_0_103_103_10s.wav", "CaMN (Upper only)", True, True, True],
    ["./examples/audio/2_scott_0_103_103_10s.wav", "EMAGE (Full body + Face)", True, True, True],
]

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
          <div>
            <h1>EMAGE</h1>
            <span>Generating Face and Body Animation from Speech</span>
            <br>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
              <a href="https://github.com/PantoMatrix/PantoMatrix"><img src="https://img.shields.io/badge/Project_Page-EMAGE-orange" alt="Project Page"></a>
              &nbsp;
              <a href="https://github.com/PantoMatrix/PantoMatrix"><img src="https://img.shields.io/badge/Github-Code-green"></a>
              &nbsp;
              <a href="https://github.com/PantoMatrix/PantoMatrix"><img src="https://img.shields.io/github/stars/PantoMatrix/PantoMatrix" alt="Stars"></a>
            </div>
          </div>
        </div>
        """
        )
        with gr.Row():
            input_audio = gr.Audio(type="numpy", label="Upload Audio")
            with gr.Column():
                model_type = gr.Radio(
                    choices=["DisCo (Upper only)", "CaMN (Upper only)", "EMAGE (Full body + Face)"],
                    value="CaMN (Upper only)",
                    label="Select Model: DisCo/CaMN for Upper, EMAGE for Full Body+Face"
                )
                render_face = gr.Checkbox(value=False, label="Render 2D Face Landmark (Fast ~4s for 7s)")
                render_mesh = gr.Checkbox(value=False, label="Render Mesh Body (Slow ~1min for 7s)")
                render_mesh_face = gr.Checkbox(value=False, label="Render Mesh Face (Extra Slow)")

        btn = gr.Button("Run Inference")

    with gr.Row():
        vid_body = gr.Video(label="2D Body Video")
        vid_mesh = gr.Video(label="Mesh Body Video (optional)")
        vid_face = gr.Video(label="2D Face Video (optional)")
        vid_meshface = gr.Video(label="Mesh Face Video (optional)")

    with gr.Column():
        gr.Markdown("Download Motion NPZ, Use Our [Blender Add-on](https://huggingface.co/datasets/H-Liu1997/BEAT2_Tools/blob/main/smplx_blender_addon_20230921.zip) for Visualization. [Demo](https://github.com/PantoMatrix/PantoMatrix/issues/178) of how to install on blender.")
        file_npz = gr.File(label="Motion NPZ")

    btn.click(
        fn=inference_app,
        inputs=[input_audio, model_type, render_mesh, render_face, render_mesh_face],
        outputs=[vid_body, vid_mesh, vid_face, vid_meshface, file_npz]
    )

    gr.Examples(
      examples=examples_data,
      inputs=[input_audio, model_type, render_mesh, render_face, render_mesh_face],
      outputs=[vid_body, vid_mesh, vid_face, vid_meshface, file_npz],
      fn=inference_app,
      cache_examples=True
      )

if __name__ == "__main__":
    demo.launch(share=True)
