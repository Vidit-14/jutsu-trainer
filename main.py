import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import time, math, random, pygame, threading, os
from flask import Flask, Response, jsonify, render_template
from collections import deque
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════
state_lock = threading.Lock()
STATE = {
    "current_sign": None, "conf": 0.0, "progress": [],
    "active_jutsu": None, "sharingan_active": False,
    "mangekyou": False, "fps": 0.0, "jutsu_count": 0,
    "control_scale": 1.0, "control_variant": 0.0,
    "_reset": False,
}
_frame_queue = deque(maxlen=1)

# ═══════════════════════════════════════════════════════════════
#  BUFFER POOL
# ═══════════════════════════════════════════════════════════════
_BUF_POOL = {}
def get_buf(shape):
    key = shape
    if key not in _BUF_POOL:
        _BUF_POOL[key] = [np.zeros(shape, dtype=np.uint8) for _ in range(8)]
        _BUF_POOL[key+('idx',)] = 0
    idx = _BUF_POOL[key+('idx',)]
    buf = _BUF_POOL[key][idx % 8]; buf[:] = 0
    _BUF_POOL[key+('idx',)] = idx+1
    return buf

# ═══════════════════════════════════════════════════════════════
#  SOUND
# ═══════════════════════════════════════════════════════════════
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
SOUNDS_DIR = "sounds"
def _make_snd(wave):
    wave = np.clip(wave*32767,-32767,32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([wave,wave]))
def _load_or_gen(fn, gfn):
    p=os.path.join(SOUNDS_DIR,fn)
    return pygame.mixer.Sound(p) if os.path.exists(p) else gfn()
def _gen_fireball():
    sr=44100;t=np.linspace(0,1.5,int(sr*1.5),False)
    return _make_snd((np.random.uniform(-1,1,len(t))*0.6+np.sin(2*np.pi*55*t)*0.4)*np.exp(-t*1.5)*0.85)
def _gen_chidori():
    sr=44100;t=np.linspace(0,1.4,int(sr*1.4),False)
    freq=1200+3000*np.abs(np.sin(2*np.pi*22*t));w=np.sin(2*np.pi*freq*t)*0.55+np.random.uniform(-1,1,len(t))*0.40
    env=np.ones(len(t));env[:300]=np.linspace(0,1,300);env[-800:]=np.linspace(1,0,800)
    return _make_snd(w*env*0.75)
def _gen_rasengan():
    sr=44100;t=np.linspace(0,1.2,int(sr*1.2),False)
    w=np.sin(2*np.pi*np.linspace(60,900,len(t))*t)*0.6+np.random.uniform(-1,1,len(t))*0.3
    env=np.ones(len(t));env[:int(sr*0.12)]=np.linspace(0,1,int(sr*0.12));env[int(sr*0.85):]=np.linspace(1,0,len(t)-int(sr*0.85))
    return _make_snd(w*env*0.75)
def _gen_shadow():
    sr=44100;n=int(sr*0.6);t=np.linspace(0,0.6,n,False)
    return _make_snd(np.random.uniform(-1,1,n)*np.exp(-t*6)*0.7)
def _gen_sharingan():
    sr=44100;t=np.linspace(0,1.2,int(sr*1.2),False)
    return _make_snd((np.sin(2*np.pi*180*t)*0.35+np.sin(2*np.pi*360*t)*0.28+np.sin(2*np.pi*720*t)*0.20)*np.exp(-t*1.4)*0.70)
def _gen_flamethrower():
    sr=44100;n=sr;t=np.linspace(0,1.0,n,False)
    w=np.random.uniform(-1,1,n)*0.75+np.sin(2*np.pi*65*t)*0.25
    env=np.ones(n);env[:int(sr*0.07)]=np.linspace(0,1,int(sr*0.07));env[int(sr*0.82):]=np.linspace(1,0,n-int(sr*0.82))
    return _make_snd(w*env*0.85)
def _gen_susanoo():
    sr=44100;n=sr*2;t=np.linspace(0,2.0,n,False)
    w=np.sin(2*np.pi*32*t)*0.5+np.sin(2*np.pi*64*t)*0.3+np.sin(2*np.pi*128*t)*0.15+np.random.uniform(-1,1,n)*0.10
    env=np.ones(n);env[:int(sr*0.35)]=np.linspace(0,1,int(sr*0.35));env[int(sr*1.65):]=np.linspace(1,0,n-int(sr*1.65))
    return _make_snd(w*env*0.88)
def _gen_kirin():
    sr=44100;n=int(sr*2.0);t=np.linspace(0,2.0,n,False)
    crack_len=int(sr*0.08);crack=np.random.uniform(-1,1,crack_len)*np.linspace(1,0,crack_len)
    rumble_t=np.linspace(0,1.92,n-crack_len)
    rumble=(np.random.uniform(-1,1,n-crack_len)*0.6+np.sin(2*np.pi*38*rumble_t)*0.3)*np.exp(-rumble_t*2.5)
    w=np.concatenate([crack,rumble])+np.random.uniform(-1,1,n)*0.25*np.exp(-t*3)
    return _make_snd(w*0.90)
def _gen_water_dragon():
    sr=44100;n=int(sr*1.5);t=np.linspace(0,1.5,n,False)
    w=np.random.uniform(-1,1,n)*0.5+np.sin(2*np.pi*48*t)*0.35+np.sin(2*np.pi*96*t)*0.20
    env=np.ones(n);env[:int(sr*0.1)]=np.linspace(0,1,int(sr*0.1));env[int(sr*1.1):]=np.linspace(1,0,n-int(sr*1.1))
    return _make_snd(w*env*0.80)
def _gen_amaterasu():
    sr=44100;n=int(sr*1.5);t=np.linspace(0,1.5,n,False)
    w=np.random.uniform(-1,1,n)*0.6*0.5+np.sin(2*np.pi*22*t)*0.40+np.sin(2*np.pi*44*t)*0.20
    env=np.ones(n);env[:int(sr*0.05)]=np.linspace(0,1,int(sr*0.05))
    return _make_snd(w*env*0.85)
def _gen_sand_shield():
    sr=44100;n=int(sr*1.2);t=np.linspace(0,1.2,n,False)
    w=np.random.uniform(-1,1,n)*0.6*(0.5+0.5*np.sin(2*np.pi*5*t))+np.sin(2*np.pi*70*t)*0.3*np.exp(-t*4)
    env=np.ones(n);env[:int(sr*0.06)]=np.linspace(0,1,int(sr*0.06));env[int(sr*0.9):]=np.linspace(1,0,n-int(sr*0.9))
    return _make_snd(w*env*0.80)
def _gen_mangekyou():
    sr=44100;t=np.linspace(0,1.5,int(sr*1.5),False)
    freq=280+420*np.exp(-t*2)
    w=(np.sin(2*np.pi*freq*t)*0.45+np.sin(2*np.pi*freq*2.5*t)*0.25+np.sin(2*np.pi*freq*0.5*t)*0.20)
    env=np.ones(len(t));env[:int(sr*0.05)]=np.linspace(0,1,int(sr*0.05));env[int(sr*1.1):]=np.linspace(1,0,len(t)-int(sr*1.1))
    return _make_snd(w*env*0.80)

os.makedirs(SOUNDS_DIR, exist_ok=True)
print("Loading sounds...")
SOUNDS={
    "fireball":_load_or_gen("fireball.wav",_gen_fireball),
    "chidori":_load_or_gen("chidori.wav",_gen_chidori),
    "rasengan":_load_or_gen("rasengan.wav",_gen_rasengan),
    "shadow_clone":_load_or_gen("shadow_clone.wav",_gen_shadow),
    "sharingan":_load_or_gen("sharingan.wav",_gen_sharingan),
    "flamethrower":_load_or_gen("flamethrower.wav",_gen_flamethrower),
    "susanoo":_load_or_gen("susanoo.wav",_gen_susanoo),
    "kirin":_load_or_gen("kirin.wav",_gen_kirin),
    "water_dragon":_load_or_gen("water_dragon.wav",_gen_water_dragon),
    "amaterasu":_load_or_gen("amaterasu.wav",_gen_amaterasu),
    "sand_shield":_load_or_gen("sand_shield.wav",_gen_sand_shield),
    "mangekyou":_load_or_gen("mangekyou.wav",_gen_mangekyou),
}
def play_sound(name):
    if name in SOUNDS: SOUNDS[name].stop(); SOUNDS[name].play()

# ═══════════════════════════════════════════════════════════════
#  CNN MODEL
# ═══════════════════════════════════════════════════════════════
class JutsuCNN(nn.Module):
    def __init__(self,num_classes=12):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout2d(0.25),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout2d(0.25),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout2d(0.25),
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),nn.Linear(128*8*8,512),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(512,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,num_classes)
        )
    def forward(self,x): return self.classifier(self.features(x))

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint=torch.load("jutsu_cnn.pth",map_location=device)
SIGNS=checkpoint["signs"]; IMG_SIZE=checkpoint["img_size"]
cnn_model=JutsuCNN(num_classes=len(SIGNS)).to(device)
cnn_model.load_state_dict(checkpoint["model_state"]); cnn_model.eval()
preprocess=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
PADDING=40
print(f"CNN loaded — {len(SIGNS)} signs — {device}")

# ═══════════════════════════════════════════════════════════════
#  MEDIAPIPE
# ═══════════════════════════════════════════════════════════════
BaseOptions=mp.tasks.BaseOptions; VisionRunningMode=mp.tasks.vision.RunningMode
HandLandmarker=mp.tasks.vision.HandLandmarker; HandLandmarkerOpts=mp.tasks.vision.HandLandmarkerOptions
FaceLandmarker=mp.tasks.vision.FaceLandmarker; FaceLandmarkerOpts=mp.tasks.vision.FaceLandmarkerOptions
hand_options=HandLandmarkerOpts(base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=0.5,min_tracking_confidence=0.5)
face_options=FaceLandmarkerOpts(base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,num_faces=1,min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,min_tracking_confidence=0.5)

# ═══════════════════════════════════════════════════════════════
#  JUTSU DATA
# ═══════════════════════════════════════════════════════════════
JUTSU_SEQUENCES={
    "fireball":["horse","tiger"],"chidori":["ox","hare","monkey","bird","dog"],
    "rasengan":["ram","snake","horse"],"shadow_clone":["ram","snake","tiger"],
    "sharingan":["tiger","boar","ox","dog"],"flamethrower":["horse","dragon","tiger"],
    "susanoo":["dragon","bird","ox"],"water_dragon":["dragon","ram","bird"],
    "amaterasu":["dog","tiger","boar"],"sand_shield":["dog","boar","ram"],
}
COMBOS={"chidori":("sharingan","kirin"),"amaterasu":("sharingan","amaterasu")}
JUTSU_NAMES={
    "fireball":"KATON: GOKAKYUU NO JUTSU!","chidori":"CHIDORI!","rasengan":"RASENGAN!",
    "shadow_clone":"KAGE BUNSHIN NO JUTSU!","sharingan":"SHARINGAN!",
    "flamethrower":"KATON: RYUUKA NO JUTSU!","susanoo":"SUSANOO!","kirin":"KIRIN!",
    "water_dragon":"SUITON: SUIRYUUDAN!","amaterasu":"AMATERASU!",
    "sand_shield":"SABAKU KYU!","mangekyou":"MANGEKYOU SHARINGAN!",
}
JUTSU_COLORS_BGR={
    "fireball":(0,80,255),"chidori":(255,220,50),"rasengan":(255,200,100),
    "shadow_clone":(0,215,255),"sharingan":(0,0,255),"flamethrower":(0,50,255),
    "susanoo":(255,100,50),"kirin":(200,230,255),"water_dragon":(255,180,0),
    "amaterasu":(60,0,0),"sand_shield":(0,160,210),"mangekyou":(200,0,255),
}

# ═══════════════════════════════════════════════════════════════
#  HAND IDENTITY ANCHOR
#
#  MediaPipe's hand indices (0/1) can swap arbitrarily between
#  frames whenever hands cross or briefly leave frame. Instead of
#  trusting the index, we store the WRIST PIXEL POSITION of the
#  effect hand at trigger time and re-identify it every frame by
#  proximity. The closest wrist to the anchor = effect hand;
#  the other = control hand. We also slide the anchor each frame
#  to follow the effect hand smoothly.
# ═══════════════════════════════════════════════════════════════
class HandAnchor:
    def __init__(self):
        self.effect_wrist = None  # (x, y) pixel anchor
        self.locked = False

    def lock(self, wrist_px):
        self.effect_wrist = wrist_px
        self.locked = True

    def unlock(self):
        self.effect_wrist = None
        self.locked = False

    def resolve(self, hand_result, fw, fh):
        """
        Returns (effect_pts, control_pts).
        Each is a list of 21 (x,y) pixel tuples, or None if absent.
        Matching is by wrist proximity — never by MediaPipe index.
        """
        if not hand_result or not hand_result.hand_landmarks:
            return None, None

        hands = [[(int(lm.x*fw), int(lm.y*fh)) for lm in hand]
                 for hand in hand_result.hand_landmarks]

        if not self.locked or self.effect_wrist is None:
            return (hands[0] if hands else None,
                    hands[1] if len(hands) > 1 else None)

        ex, ey = self.effect_wrist
        best_dist = float('inf')
        effect_idx = 0
        for i, pts in enumerate(hands):
            wx, wy = pts[0]
            d = math.hypot(wx - ex, wy - ey)
            if d < best_dist:
                best_dist = d
                effect_idx = i

        # Slide anchor to keep following the effect hand
        self.effect_wrist = hands[effect_idx][0]

        return (hands[effect_idx],
                hands[1 - effect_idx] if len(hands) > 1 else None)


# ═══════════════════════════════════════════════════════════════
#  WRIST ROLL  (pose-invariant)
#
#  We measure the roll of the control hand's wrist by projecting
#  the cross-hand axis (pinky-MCP→index-MCP, lm 17→5) onto the
#  plane perpendicular to the forward axis (wrist→mid-MCP, lm 0→9).
#  This angle only changes when the wrist physically rotates —
#  moving or pointing the hand in a different direction has no
#  effect.
# ═══════════════════════════════════════════════════════════════
def get_wrist_roll(pts):
    """Return wrist roll as a value in [0, 1]."""
    wrist    = np.array(pts[0],  dtype=float)
    mid_mcp  = np.array(pts[9],  dtype=float)
    idx_mcp  = np.array(pts[5],  dtype=float)
    pink_mcp = np.array(pts[17], dtype=float)

    # Forward axis: wrist → mid-MCP
    fwd = mid_mcp - wrist
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm < 1e-6: return 0.0
    fwd /= fwd_norm

    # Side axis: pinky-MCP → index-MCP
    side = idx_mcp - pink_mcp
    side_norm = np.linalg.norm(side)
    if side_norm < 1e-6: return 0.0
    side /= side_norm

    # Remove any component along forward (project onto perpendicular plane)
    side_perp = side - np.dot(side, fwd) * fwd
    sp_norm = np.linalg.norm(side_perp)
    if sp_norm < 1e-6: return 0.0
    side_perp /= sp_norm

    # Fixed reference: screen "up" = negative Y direction
    up = np.array([0.0, -1.0])

    # Project up onto the same plane
    ref = up - np.dot(up, fwd) * fwd
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-6:
        # Hand pointing straight up — use x-axis as fallback
        ref = np.array([1.0, 0.0])
        ref = ref - np.dot(ref, fwd) * fwd
        ref_norm = np.linalg.norm(ref)
        if ref_norm < 1e-6: return 0.0
    ref /= ref_norm

    # Signed angle between side_perp and ref using 2-D cross product
    cos_a = float(np.clip(np.dot(side_perp, ref), -1.0, 1.0))
    cross  = float(side_perp[0]*ref[1] - side_perp[1]*ref[0])
    angle  = math.atan2(cross, cos_a)   # [-π, π]
    return (angle + math.pi) / (2 * math.pi)   # → [0, 1]


def get_pinch_scale(pts):
    """Thumb-tip (4) ↔ index-tip (8) distance → scale in [0.4, 2.0]."""
    d = math.hypot(pts[4][0]-pts[8][0], pts[4][1]-pts[8][1])
    return max(0.4, min(2.0, 0.4 + (d - 20) / (150 - 20) * 1.6))


# ═══════════════════════════════════════════════════════════════
#  OTHER HELPERS
# ═══════════════════════════════════════════════════════════════
def get_both_hands_bbox(result, fw, fh):
    if not result or not result.hand_landmarks: return None
    xs, ys = [], []
    for hand in result.hand_landmarks:
        for lm in hand: xs.append(int(lm.x*fw)); ys.append(int(lm.y*fh))
    x1=max(0,min(xs)-PADDING); y1=max(0,min(ys)-PADDING)
    x2=min(fw,max(xs)+PADDING); y2=min(fh,max(ys)+PADDING)
    if (x2-x1)<20 or (y2-y1)<20: return None
    return x1,y1,x2,y2

def pts_center(pts):
    return int(np.mean([p[0] for p in pts])), int(np.mean([p[1] for p in pts]))

def get_face_data(face_result,w,h):
    if not face_result or not face_result.face_landmarks: return None
    lm=face_result.face_landmarks[0]
    def pt(i): return int(lm[i].x*w),int(lm[i].y*h)
    left_eye=pt(468); right_eye=pt(473)
    left_r=max(5,int(math.hypot(lm[469].x*w-lm[468].x*w,lm[469].y*h-lm[468].y*h)))
    right_r=max(5,int(math.hypot(lm[474].x*w-lm[473].x*w,lm[474].y*h-lm[473].y*h)))
    mx=int((lm[61].x+lm[291].x)/2*w); my=int((lm[13].y+lm[14].y)/2*h)
    mouth_w=int(abs(lm[291].x-lm[61].x)*w)
    face_cx=int(lm[1].x*w); face_cy=int((lm[10].y+lm[152].y)/2*h)
    face_h=int(abs(lm[152].y-lm[10].y)*h); face_w=int(abs(lm[454].x-lm[234].x)*w)
    return {"left_eye":left_eye,"left_r":left_r,"right_eye":right_eye,"right_r":right_r,
            "mouth":(mx,my),"mouth_w":mouth_w,"face_cx":face_cx,"face_cy":face_cy,"face_h":face_h,"face_w":face_w}

def clip_pt(x,y,w,h): return int(max(0,min(w-1,x))),int(max(0,min(h-1,y)))

def fast_glow(frame,cx,cy,radius,color,strength):
    if radius<2: return
    buf=get_buf(frame.shape); cv2.circle(buf,(int(cx),int(cy)),radius,color,-1)
    fh,fw=buf.shape[:2]; ds=4
    small=cv2.resize(buf,(max(1,fw//ds),max(1,fh//ds))); k=max(1,(radius//(ds*2))|1)
    small=cv2.GaussianBlur(small,(k,k),0)
    cv2.addWeighted(cv2.resize(small,(fw,fh)),strength,frame,1.0,0,frame)

def variant_color(base_bgr, variant):
    arr=np.uint8([[list(base_bgr)]])
    hsv=cv2.cvtColor(arr,cv2.COLOR_BGR2HSV)[0][0]
    hsv[0]=int((int(hsv[0])+int(variant*180))%180)
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[hsv]]),cv2.COLOR_HSV2BGR)[0][0])

# ═══════════════════════════════════════════════════════════════
#  THREADED DETECTORS
# ═══════════════════════════════════════════════════════════════
class DetectorThread(threading.Thread):
    def __init__(self,options_obj,model_class):
        super().__init__(daemon=True)
        self._options=options_obj; self._model_cls=model_class
        self._frame=None; self._result=None
        self._frame_lock=threading.Lock(); self._res_lock=threading.Lock()
        self._new_frame=threading.Event(); self.running=True
    def push_frame(self,frame):
        with self._frame_lock: self._frame=frame
        self._new_frame.set()
    def get_result(self):
        with self._res_lock: return self._result
    def run(self):
        with self._model_cls.create_from_options(self._options) as lm:
            while self.running:
                if not self._new_frame.wait(timeout=0.05): continue
                self._new_frame.clear()
                with self._frame_lock: frame=self._frame
                if frame is None: continue
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                result=lm.detect(mp_img)
                with self._res_lock: self._result=result
    def stop(self): self.running=False

class CNNThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._crop=None; self._result=(None,0.0)
        self._lock=threading.Lock(); self._new=threading.Event(); self.running=True
    def push_data(self,frame,bbox):
        x1,y1,x2,y2=bbox
        with self._lock: self._crop=frame[y1:y2,x1:x2].copy()
        self._new.set()
    def get_result(self):
        with self._lock: return self._result
    def run(self):
        while self.running:
            if not self._new.wait(timeout=0.05): continue
            self._new.clear()
            with self._lock: crop=self._crop
            if crop is None or crop.size==0: continue
            gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4)); gray=clahe.apply(gray)
            tensor=preprocess(Image.fromarray(gray)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs=torch.softmax(cnn_model(tensor),dim=1)[0]; conf,idx=probs.max(0)
            with self._lock: self._result=(SIGNS[idx.item()],conf.item())
    def stop(self): self.running=False

class PredictionSmoother:
    def __init__(self,window=8,conf_threshold=0.70,vote_threshold=0.55):
        self.window=window; self.conf_threshold=conf_threshold
        self.vote_threshold=vote_threshold; self.history=deque(maxlen=window)
    def update(self,sign,conf):
        self.history.append(sign if (sign is not None and conf>=self.conf_threshold) else None)
        if len(self.history)<self.window: return None
        valid=[p for p in self.history if p is not None]
        if not valid: return None
        counts={}
        for p in valid: counts[p]=counts.get(p,0)+1
        best=max(counts,key=counts.get)
        return best if counts[best]/self.window>=self.vote_threshold else None
    def reset(self): self.history.clear()

class SequenceEngine:
    def __init__(self,sequences,hold_frames=8):
        self.sequences=sequences; self.hold_frames=hold_frames
        self.buffer=deque(maxlen=20); self.last_sign=None; self.hold_count=0
    def update(self,sign):
        if sign==self.last_sign: self.hold_count+=1
        else: self.last_sign=sign; self.hold_count=1
        if self.hold_count==self.hold_frames:
            if not self.buffer or self.buffer[-1]!=sign:
                self.buffer.append(sign); print(f"sign: {sign.upper()}")
        buf=list(self.buffer)
        for jutsu,seq in self.sequences.items():
            n=len(seq)
            if len(buf)>=n and buf[-n:]==seq:
                self.buffer.clear(); return jutsu
        return None
    def get_progress(self): return list(self.buffer)
    def reset(self): self.buffer.clear(); self.last_sign=None; self.hold_count=0

class ScreenEffects:
    def __init__(self):
        self.flash_alpha=0.0; self.flash_color=(255,255,255)
        self.shake_frames=0; self.shake_power=0; self._white=None
    def trigger(self,color,flash_color=(255,255,255),shake=14):
        self.flash_alpha=1.0; self.flash_color=flash_color
        self.shake_frames=18; self.shake_power=shake
    def apply(self,frame):
        h,w=frame.shape[:2]
        if self.shake_frames>0:
            intensity=int(self.shake_power*(self.shake_frames/18))
            dx=random.randint(-intensity,intensity); dy=random.randint(-intensity,intensity)
            frame=cv2.warpAffine(frame,np.float32([[1,0,dx],[0,1,dy]]),(w,h))
            self.shake_frames-=1
        if self.flash_alpha>0.01:
            if self._white is None or self._white.shape!=frame.shape:
                self._white=np.full(frame.shape,255,dtype=np.uint8)
            self._white[:]=self.flash_color
            cv2.addWeighted(self._white,self.flash_alpha,frame,1.0-self.flash_alpha,0,frame)
            self.flash_alpha*=0.68
        return frame

# ═══════════════════════════════════════════════════════════════
#  EFFECTS
# ═══════════════════════════════════════════════════════════════
class FireballEffect:
    def __init__(self): self.active=False; self.particles=[]; self.embers=[]; self.t=0; self.cx=self.cy=0
    def start(self,cx,cy): self.active=True; self.particles=[]; self.embers=[]; self.t=0; self.cx=cx; self.cy=cy
    def stop(self): self.active=False; self.particles=[]; self.embers=[]
    def _spawn(self,scale,variant):
        for _ in range(int(18*scale)):
            a=random.uniform(0,2*math.pi); s=random.uniform(2,9)*scale
            self.particles.append({"x":self.cx+random.uniform(-10,10),"y":self.cy+random.uniform(-10,10),"vx":math.cos(a)*s,"vy":math.sin(a)*s-random.uniform(1,3),"life":random.uniform(20,40),"max_life":40,"size":random.uniform(5,16)*scale,"layer":random.randint(0,2)})
        for _ in range(3):
            a=random.uniform(-math.pi,0)
            self.embers.append({"x":self.cx+random.uniform(-30,30),"y":self.cy+random.uniform(-10,10),"vx":math.cos(a)*random.uniform(1,4),"vy":math.sin(a)*random.uniform(2,6),"life":random.uniform(40,70),"max_life":70})
    def update(self,frame,cx,cy,scale=1.0,variant=0.0):
        if not self.active: return
        self.cx=cx; self.cy=cy; self.t+=1; self._spawn(scale,variant)
        fast_glow(frame,cx,cy,int(110*scale),(0,30,120),0.35)
        fast_glow(frame,cx,cy,int(70*scale),(0,80,200),0.40)
        fast_glow(frame,cx,cy,int(45*scale),(0,150,255),0.45)
        fast_glow(frame,cx,cy,int(22*scale),(40,210,255),0.52)
        if self.t%3==0:
            r=int(120*scale); rx1=max(0,int(cx)-r); rx2=min(frame.shape[1],int(cx)+r)
            ry1=max(0,int(cy)-r); ry2=min(frame.shape[0],int(cy)+r)
            roi=frame[ry1:ry2,rx1:rx2]
            if roi.size>0: frame[ry1:ry2,rx1:rx2]=cv2.addWeighted(cv2.GaussianBlur(roi,(5,5),0),0.18,roi,0.82,0)
        COLS=[[(200,240,255),(255,255,255)],[(0,140,255),(0,200,255)],[(0,50,200),(0,80,180)]]
        p_layer=get_buf(frame.shape); alive=[]
        for p in self.particles:
            p["x"]+=p["vx"]; p["y"]+=p["vy"]; p["vy"]-=0.18; p["life"]-=1
            if p["life"]>0:
                a=p["life"]/p["max_life"]; sz=max(1,int(p["size"]*a))
                cols=COLS[p["layer"]]; base_col=tuple(int(c*a) for c in (cols[0] if a>0.5 else cols[1]))
                color=variant_color(base_col,variant)
                fh2,fw2=frame.shape[:2]; px=int(p["x"]); py=int(p["y"])
                if 0<=px<fw2 and 0<=py<fh2: cv2.circle(p_layer,(px,py),sz,color,-1)
                alive.append(p)
        self.particles=alive; alive_e=[]
        for e in self.embers:
            e["x"]+=e["vx"]; e["y"]+=e["vy"]; e["vy"]-=0.05; e["vx"]+=random.uniform(-0.2,0.2); e["life"]-=1
            if e["life"]>0:
                a=e["life"]/e["max_life"]; fh2,fw2=frame.shape[:2]; px=int(e["x"]); py=int(e["y"])
                if 0<=px<fw2 and 0<=py<fh2:
                    cv2.circle(p_layer,(px,py),max(1,int(2*a)),variant_color((0,int(140*a),int(255*a)),variant),-1)
                alive_e.append(e)
        self.embers=alive_e; cv2.add(frame,p_layer,dst=frame)

class ChidoriEffect:
    def __init__(self): self.active=False; self.t=0; self.cx=self.cy=0
    def start(self,cx,cy): self.active=True; self.t=0; self.cx=cx; self.cy=cy
    def stop(self): self.active=False
    def _jagged(self,layer,x1,y1,x2,y2,depth,color,thickness):
        h,w=layer.shape[:2]
        if depth==0 or math.hypot(x2-x1,y2-y1)<4:
            cv2.line(layer,clip_pt(x1,y1,w,h),clip_pt(x2,y2,w,h),color,thickness,cv2.LINE_AA); return
        mx=(x1+x2)/2+random.uniform(-20,20); my=(y1+y2)/2+random.uniform(-20,20)
        self._jagged(layer,x1,y1,mx,my,depth-1,color,thickness); self._jagged(layer,mx,my,x2,y2,depth-1,color,thickness)
    def update(self,frame,cx,cy,scale=1.0,variant=0.0):
        if not self.active: return
        self.cx=cx; self.cy=cy; self.t+=1; h,w=frame.shape[:2]
        arc_col=variant_color((255,220,50),variant)
        fast_glow(frame,cx,cy,int(200*scale),(60,50,10),0.28)
        fast_glow(frame,cx,cy,int(80*scale),(200,150,30),0.45)
        fast_glow(frame,cx,cy,int(18*scale),(255,255,255),0.75)
        l_layer=get_buf(frame.shape); n_arcs=int(10*scale)
        for i in range(n_arcs):
            angle=(i/n_arcs)*2*math.pi+self.t*0.22+random.uniform(-0.4,0.4); length=random.uniform(60,160)*scale
            ex=cx+math.cos(angle)*length; ey=cy+math.sin(angle)*length
            self._jagged(l_layer,cx,cy,ex,ey,4,arc_col,4)
            self._jagged(l_layer,cx,cy,ex,ey,2,(255,255,255),1)
        for corner in random.sample([(0,0),(w,0),(0,h),(w,h)],2):
            if random.random()<0.4: self._jagged(l_layer,corner[0],corner[1],cx,cy,4,(100,80,20),2)
        for _ in range(int(12*scale)):
            a=random.uniform(0,2*math.pi); d=random.uniform(20,110)*scale
            sx=int(cx+math.cos(a)*d); sy=int(cy+math.sin(a)*d)
            if 0<=sx<w and 0<=sy<h: cv2.circle(l_layer,(sx,sy),random.randint(1,4),arc_col,-1)
        cv2.add(frame,l_layer,dst=frame)

class RasenganEffect:
    def __init__(self): self.active=False; self.angle=0.0; self.cx=self.cy=0; self.wind=[]
    def start(self,cx,cy): self.active=True; self.angle=0.0; self.cx=cx; self.cy=cy; self.wind=[]
    def stop(self): self.active=False; self.wind=[]
    def update(self,frame,cx,cy,scale=1.0,variant=0.0):
        if not self.active: return
        self.cx=cx; self.cy=cy; self.angle+=0.14*(1+variant*0.5); h,w=frame.shape[:2]
        if len(self.wind)<150:
            for _ in range(6):
                a=self.angle+random.uniform(0,2*math.pi); s=random.uniform(4,9)*scale
                self.wind.append({"x":self.cx+math.cos(a)*20,"y":self.cy+math.sin(a)*20,"vx":math.cos(a+math.pi/2)*s,"vy":math.sin(a+math.pi/2)*s,"life":random.uniform(10,22),"max_life":22})
        pulse=0.5+0.5*math.sin(self.angle*8)
        base_col=variant_color((255,200,100),variant)
        fast_glow(frame,cx,cy,int(85*scale),base_col,0.35)
        fast_glow(frame,cx,cy,int(50*scale),(255,230,140),0.50)
        fast_glow(frame,cx,cy,int((18+pulse*6)*scale),(255,255,255),0.78)
        p_layer=get_buf(frame.shape)
        for ring in range(4):
            radius=int((30+ring*22)*scale); count=10+ring*6; speed=1.2+ring*0.5+variant; direction=1 if ring%2==0 else -1
            for i in range(count):
                a=(i/count)*2*math.pi+self.angle*speed*direction; t=ring/4
                col=variant_color((int(255*(1-t*0.4)),int(230*(1-t*0.5)),int(60+195*t)),variant)
                px=int(cx+math.cos(a)*radius); py=int(cy+math.sin(a)*radius)
                if 0<=px<w and 0<=py<h: cv2.circle(p_layer,(px,py),max(1,5-ring),col,-1)
        for spiral in range(2):
            pts=[]
            for i in range(50):
                t=i/50; r=int((10+t*90)*scale); a=self.angle*5+t*3.5*math.pi+spiral*math.pi
                pts.append([max(0,min(w-1,int(cx+math.cos(a)*r))),max(0,min(h-1,int(cy+math.sin(a)*r)))])
            if len(pts)>1:
                for i in range(len(pts)-1):
                    t=i/50; col=variant_color((int(255*(1-t*0.5)),int(220*(1-t*0.3)),int(80+175*t)),variant)
                    cv2.line(p_layer,tuple(pts[i]),tuple(pts[i+1]),col,max(1,int(3*(1-t))))
        alive_w=[]
        for wp in self.wind:
            wp["x"]+=wp["vx"]; wp["y"]+=wp["vy"]; wp["life"]-=1
            if wp["life"]>0:
                a=wp["life"]/wp["max_life"]; col=variant_color((int(200*a),int(230*a),int(255*a)),variant)
                sx=int(wp["x"]); sy=int(wp["y"])
                if 0<=sx<w and 0<=sy<h:
                    cv2.line(p_layer,(sx,sy),(max(0,min(w-1,int(wp["x"]+wp["vx"]*2))),max(0,min(h-1,int(wp["y"]+wp["vy"]*2)))),col,1)
                alive_w.append(wp)
        self.wind=alive_w; cv2.add(frame,p_layer,dst=frame)

class ShadowCloneEffect:
    def __init__(self):
        self.active=False; self.clones=[]; self.current=0
        self.last_spawn=0; self.interval=0.45
        self.snapshot=None; self.smoke_particles=[]; self.t=0
    def start(self,frame):
        self.active=True; self.t=0; self.snapshot=frame.copy()
        self.clones=[]; self.current=0; self.last_spawn=time.time()-self.interval; self.smoke_particles=[]
    def stop(self): self.active=False; self.clones=[]; self.snapshot=None; self.smoke_particles=[]
    def _spawn_smoke(self,cx,cy):
        for _ in range(40):
            a=random.uniform(0,2*math.pi); s=random.uniform(2,8)
            self.smoke_particles.append({"x":float(cx)+random.uniform(-20,20),"y":float(cy)+random.uniform(-20,20),"vx":math.cos(a)*s,"vy":math.sin(a)*s-random.uniform(1,4),"life":random.uniform(20,45),"max_life":45,"size":random.uniform(8,30)})
    def update(self,frame,scale=1.0,variant=0.0):
        if not self.active or self.snapshot is None: return
        now=time.time(); h,w=frame.shape[:2]; self.t+=1
        num_clones=3
        if self.current<num_clones and now-self.last_spawn>=self.interval:
            spacing=w//(num_clones+1); cx=spacing*(self.current+1); cy=h//2
            self.clones.append({"cx":cx,"cy":cy,"born":now})
            self._spawn_smoke(cx,cy); self.current+=1; self.last_spawn=now
        cv2.addWeighted(frame,0.55,np.zeros_like(frame),0.45,0,frame)
        clone_layer=get_buf(frame.shape)
        tint_col=variant_color((200,220,255),variant)
        for clone in self.clones:
            age=now-clone["born"]; alpha=min(0.85,age*2.5)
            offset=clone["cx"]-w//2; shifted=np.zeros_like(self.snapshot)
            if offset>0: amt=min(abs(offset),w-1); shifted[:,amt:]=self.snapshot[:,:w-amt]
            elif offset<0: amt=min(abs(offset),w-1); shifted[:,:w-amt]=self.snapshot[:,amt:]
            else: shifted=self.snapshot.copy()
            tint=np.full_like(shifted,tint_col); tinted=cv2.addWeighted(shifted,0.75,tint,0.25,0)
            cv2.addWeighted(tinted,alpha*0.9,clone_layer,1.0,0,clone_layer)
        cv2.add(frame,clone_layer,dst=frame)
        p_layer=get_buf(frame.shape); alive=[]
        for p in self.smoke_particles:
            p["x"]+=p["vx"]; p["y"]+=p["vy"]; p["vy"]-=0.12; p["life"]-=1
            if p["life"]>0:
                a=p["life"]/p["max_life"]; sz=max(1,int(p["size"]*a)); brightness=int(200*a)
                px=int(p["x"]); py=int(p["y"])
                if 0<=px<w and 0<=py<h: cv2.circle(p_layer,(px,py),sz,(brightness,brightness,brightness),-1)
                alive.append(p)
        self.smoke_particles=alive
        if np.any(p_layer>0):
            blurred=cv2.GaussianBlur(p_layer,(15,15),0)
            cv2.addWeighted(blurred,0.7,frame,1.0,0,frame)
            cv2.addWeighted(p_layer,0.4,frame,1.0,0,frame)

class SharinganEffect:
    def __init__(self): self.active=False; self.angle=0.0; self.t=0; self.jutsu_count=0; self.mangekyou=False
    def start(self): self.active=True; self.angle=0.0; self.t=0
    def stop(self): self.active=False
    def on_jutsu_cast(self):
        if not self.active or self.mangekyou: return False
        self.jutsu_count+=1
        if self.jutsu_count>=3:
            self.mangekyou=True; print("MANGEKYOU AWAKENED!"); return True
        return False
    def reset_mangekyou(self): self.jutsu_count=0; self.mangekyou=False
    def _draw_tomoe(self,layer,cx,cy,radius,angle_offset,color,h,w):
        for i in range(3):
            a=angle_offset+i*(2*math.pi/3); hr=radius*0.52
            hx=max(0,min(w-1,int(cx+hr*math.cos(a)))); hy=max(0,min(h-1,int(cy+hr*math.sin(a))))
            cv2.circle(layer,(hx,hy),max(1,int(radius*0.22)),color,-1)
            tail=[]
            for j in range(14):
                t=j/13; ta=a+t*1.65; td=radius*(0.52-t*0.20)
                tail.append([max(0,min(w-1,int(cx+td*math.cos(ta)))),max(0,min(h-1,int(cy+td*math.sin(ta))))])
            if len(tail)>1: cv2.polylines(layer,[np.array(tail,np.int32)],False,color,max(1,int(radius*0.13)),cv2.LINE_AA)
    def _draw_mangekyou(self,layer,cx,cy,radius,angle_offset,color,h,w):
        for i in range(3):
            a=angle_offset+i*(2*math.pi/3); blade_pts=[]
            for j in range(30):
                t=j/29; sin_val=max(0.001,math.sin(t*math.pi)); r_t=radius*0.82*(sin_val**0.7)
                bx=max(0,min(w-1,int(cx+r_t*math.cos(a+t*1.2-0.6)))); by=max(0,min(h-1,int(cy+r_t*math.sin(a+t*1.2-0.6))))
                blade_pts.append([bx,by])
            if len(blade_pts)>1: cv2.polylines(layer,[np.array(blade_pts,np.int32)],False,color,max(2,int(radius*0.18)),cv2.LINE_AA)
            cv2.circle(layer,(max(0,min(w-1,int(cx+radius*0.72*math.cos(a)))),max(0,min(h-1,int(cy+radius*0.72*math.sin(a))))),max(1,int(radius*0.16)),color,-1)
    def update(self,frame,face_data):
        if not self.active or face_data is None: return
        self.angle+=0.05 if not self.mangekyou else 0.03; self.t+=1; h,w=frame.shape[:2]
        for eye_center,eye_r in [(face_data["left_eye"],face_data["left_r"]),(face_data["right_eye"],face_data["right_r"])]:
            ex,ey=eye_center; r=max(7,int(eye_r*1.05))
            ex=max(r+1,min(w-r-1,ex)); ey=max(r+1,min(h-r-1,ey))
            small_glow=get_buf(frame.shape)
            glow_col=(60,0,100) if self.mangekyou else (0,0,80)
            cv2.circle(small_glow,(ex,ey),int(r*1.6),glow_col,-1)
            cv2.GaussianBlur(small_glow,(15,15),0,small_glow)
            cv2.addWeighted(small_glow,0.28,frame,1.0,0,frame)
            eye_layer=get_buf(frame.shape)
            cv2.circle(eye_layer,(ex,ey),int(r*1.10),(0,0,0),-1)
            iris_col=(80,0,110) if self.mangekyou else (0,0,155)
            iris_layer=get_buf(frame.shape)
            cv2.circle(iris_layer,(ex,ey),r,iris_col,-1)
            cv2.addWeighted(iris_layer,0.55,eye_layer,1.0,0,eye_layer)
            if self.mangekyou: self._draw_mangekyou(eye_layer,ex,ey,r,self.angle,(0,0,0),h,w)
            else: self._draw_tomoe(eye_layer,ex,ey,r,self.angle,(0,0,0),h,w)
            cv2.circle(eye_layer,(ex,ey),max(2,int(r*0.26)),(0,0,0),-1)
            cv2.circle(eye_layer,(max(0,min(w-1,ex-int(r*0.22))),max(0,min(h-1,ey-int(r*0.22)))),max(1,int(r*0.10)),(200,190,230),-1)
            cv2.addWeighted(eye_layer,0.78,frame,1.0,0,frame)

class FlamethrowerEffect:
    def __init__(self): self.active=False; self.particles=[]; self.t=0
    def start(self): self.active=True; self.particles=[]; self.t=0
    def stop(self): self.active=False; self.particles=[]
    def update(self,frame,face_data,scale=1.0,variant=0.0):
        if not self.active or face_data is None: return
        self.t+=1; mx,my=face_data["mouth"]; mouth_w=face_data["mouth_w"]; h,w=frame.shape[:2]
        if len(self.particles)<300:
            for _ in range(int(22*scale)):
                spread=random.uniform(-math.pi/4,math.pi/4); angle=math.pi/2+spread; speed=random.uniform(5,16)*scale
                self.particles.append({"x":mx+random.uniform(-mouth_w*0.4,mouth_w*0.4),"y":my,"vx":math.cos(angle)*speed*random.uniform(0.3,1.0),"vy":math.sin(angle)*speed,"life":random.uniform(15,38),"max_life":38,"size":random.uniform(6,20)*scale,"layer":random.randint(0,2)})
        fast_glow(frame,mx,my,int(55*scale),variant_color((0,50,255),variant),0.6)
        COLS=[[(200,240,255),(255,255,200)],[(0,140,255),(0,200,255)],[(0,40,180),(0,60,150)]]; p_layer=get_buf(frame.shape); alive=[]
        for p in self.particles:
            p["x"]+=p["vx"]; p["y"]+=p["vy"]; p["vy"]+=0.25; p["vx"]+=random.uniform(-0.5,0.5); p["life"]-=1
            if p["life"]>0:
                a=p["life"]/p["max_life"]; sz=max(1,int(p["size"]*a))
                cols=COLS[p["layer"]]; color=variant_color(tuple(int(c*a) for c in (cols[0] if a>0.5 else cols[1])),variant)
                px=int(p["x"]); py=int(p["y"])
                if 0<=px<w and 0<=py<h: cv2.circle(p_layer,(px,py),sz,color,-1)
                alive.append(p)
        self.particles=alive; cv2.add(frame,p_layer,dst=frame)

class SusanooEffect:
    def __init__(self): self.active=False; self.t=0; self.orbs=[]; self.flames=[]
    def start(self): self.active=True; self.t=0; self.orbs=[]; self.flames=[]
    def stop(self): self.active=False; self.orbs=[]; self.flames=[]
    def _jagged(self,layer,x1,y1,x2,y2,depth,color,thick):
        h,w=layer.shape[:2]
        if depth==0 or math.hypot(x2-x1,y2-y1)<6: cv2.line(layer,clip_pt(x1,y1,w,h),clip_pt(x2,y2,w,h),color,thick,cv2.LINE_AA); return
        mx=(x1+x2)/2+random.uniform(-25,25); my=(y1+y2)/2+random.uniform(-25,25)
        self._jagged(layer,x1,y1,mx,my,depth-1,color,thick); self._jagged(layer,mx,my,x2,y2,depth-1,color,thick)
    def _draw_armour(self,frame,fd,color,h,w):
        if fd is None: return
        cx=fd["face_cx"]; fh=fd["face_h"]; fw=fd["face_w"]; cy=fd["face_cy"]
        layer=get_buf(frame.shape)
        neck_y=cy+int(fh*0.60); shoulder_y=cy+int(fh*0.70)
        chest_top=cy+int(fh*0.85); chest_bot=cy+int(fh*2.9); hip_y=cy+int(fh*3.1)
        for sign in [-1,1]:
            cv2.line(layer,clip_pt(cx,neck_y,w,h),clip_pt(cx+sign*int(fw*0.25),shoulder_y,w,h),color,4,cv2.LINE_AA)
        for sign in [-1,1]:
            sx=cx+sign*int(fw*1.5); sy=shoulder_y
            pauldron_pts=[]
            for j in range(20):
                t=j/19; a=math.pi*t*0.9+(math.pi if sign==-1 else 0)
                pauldron_pts.append([max(0,min(w-1,sx+sign*int(fw*0.4*math.cos(a)))),max(0,min(h-1,sy+int(fh*0.3*math.sin(a))))])
            if len(pauldron_pts)>1: cv2.polylines(layer,[np.array(pauldron_pts,np.int32)],True,color,5,cv2.LINE_AA)
            for k in range(3):
                t=k/2; a=math.pi*t*0.8+(math.pi if sign==-1 else 0)
                bx=sx+sign*int(fw*0.4*math.cos(a)); by=sy+int(fh*0.3*math.sin(a))
                cv2.line(layer,clip_pt(bx,by,w,h),clip_pt(bx+sign*int(fw*0.18),by-int(fh*0.15),w,h),color,3,cv2.LINE_AA)
        for sign in [-1,1]:
            pts=np.array([[max(0,min(w-1,cx)),max(0,min(h-1,neck_y))],[max(0,min(w-1,cx+sign*int(fw*0.5))),max(0,min(h-1,neck_y-int(fh*0.05)))],[max(0,min(w-1,cx+sign*int(fw*1.0))),max(0,min(h-1,shoulder_y))],[max(0,min(w-1,cx+sign*int(fw*1.45))),max(0,min(h-1,shoulder_y+int(fh*0.1)))]],np.int32)
            cv2.polylines(layer,[pts],False,color,5,cv2.LINE_AA); cv2.polylines(layer,[pts],False,(255,255,255),1,cv2.LINE_AA)
        for k in range(8):
            t=k/7; sy2=int(chest_top+t*(int(chest_bot*0.5+chest_top*0.5)-chest_top))
            cv2.circle(layer,(max(0,min(w-1,cx)),max(0,min(h-1,sy2))),4,color,-1)
        for i in range(8):
            rt=i/7; ry=int(chest_top+(chest_bot-chest_top)*rt)
            if ry>=h: continue
            half_w=int(fw*(0.85+rt*0.65)); curve=int(fh*0.12*rt)
            bright=int(140+115*math.sin(self.t*0.06+i*0.7))
            rc=tuple(min(255,int(c*(bright/255))) for c in color)
            for sign in [-1,1]:
                rib_pts=[]
                for j in range(22):
                    jt=j/21; rib_pts.append([max(0,min(w-1,int(cx+sign*int(half_w*jt)))),max(0,min(h-1,int(ry+curve*math.sin(jt*math.pi))))])
                arr=np.array(rib_pts,np.int32)
                cv2.polylines(layer,[arr],False,rc,5,cv2.LINE_AA); cv2.polylines(layer,[arr],False,tuple(min(255,c+70) for c in rc),1,cv2.LINE_AA)
                cv2.circle(layer,(rib_pts[-1][0],rib_pts[-1][1]),5,rc,-1)
        pelvis_pts=[]
        for j in range(20):
            t=j/19; a=math.pi+t*math.pi
            pelvis_pts.append([max(0,min(w-1,int(cx+int(fw*0.9)*math.cos(a)))),max(0,min(h-1,int(hip_y+int(fh*0.2)*math.sin(a))))])
        if len(pelvis_pts)>1: cv2.polylines(layer,[np.array(pelvis_pts,np.int32)],False,color,5,cv2.LINE_AA)
        for sign in [-1,1]:
            ax=cx+sign*int(fw*1.5); ay=shoulder_y
            for k in range(4):
                seg_y=ay+int(fh*0.4*k); seg_y2=ay+int(fh*0.4*(k+1)); seg_w=int(fw*0.22*(1-k*0.05))
                cv2.line(layer,clip_pt(ax-sign*seg_w,seg_y,w,h),clip_pt(ax-sign*seg_w,seg_y2,w,h),color,3,cv2.LINE_AA)
                cv2.line(layer,clip_pt(ax+sign*seg_w,seg_y,w,h),clip_pt(ax+sign*seg_w,seg_y2,w,h),color,3,cv2.LINE_AA)
                cv2.line(layer,clip_pt(ax-sign*seg_w,seg_y,w,h),clip_pt(ax+sign*seg_w,seg_y,w,h),color,2,cv2.LINE_AA)
        blurred=get_buf(frame.shape); cv2.GaussianBlur(layer,(7,7),0,blurred)
        cv2.addWeighted(blurred,0.7,frame,1.0,0,frame); cv2.addWeighted(layer,1.0,frame,1.0,0,frame)
    def _spawn_body_flames(self,fd,h,w):
        if fd is None: return
        cx=fd["face_cx"]; fh=fd["face_h"]; fw=fd["face_w"]; cy=fd["face_cy"]
        for side in [-1,1]:
            body_x=cx+side*int(fw*1.0)
            for _ in range(3):
                sy=random.randint(int(cy+fh*0.5),min(h-1,int(cy+fh*3.0)))
                self.flames.append({"x":float(body_x)+random.uniform(-15,15),"y":float(sy),"vx":side*random.uniform(0.5,2.5)+random.uniform(-1,1),"vy":-random.uniform(3,8),"life":random.uniform(20,45),"max_life":45,"size":random.uniform(4,16)})
    def update(self,frame,face_data,scale=1.0,variant=0.0):
        if not self.active: return
        self.t+=1; h,w=frame.shape[:2]
        pulse=0.5+0.5*math.sin(self.t*0.05)
        b_col=variant_color((int(220*pulse),int(80*(1-pulse)),int(80+120*pulse)),variant)
        aura=get_buf(frame.shape); cv2.rectangle(aura,(0,0),(w,h),(10,10,80),-1); cv2.addWeighted(aura,0.18,frame,1.0,0,frame)
        edge=get_buf(frame.shape); cv2.rectangle(edge,(0,0),(w,h),b_col,int(8+6*pulse)); cv2.addWeighted(edge,0.75,frame,1.0,0,frame)
        arc_layer=get_buf(frame.shape)
        if self.t%3==0:
            for _ in range(3): self._jagged(arc_layer,random.randint(0,w),random.randint(0,h),random.randint(0,w),random.randint(0,h),4,b_col,2)
        cv2.add(frame,arc_layer,dst=frame)
        if len(self.orbs)<20:
            for _ in range(2): self.orbs.append({"x":random.uniform(0,w),"y":random.uniform(0,h),"vx":random.uniform(-2,2),"vy":random.uniform(-2,2),"life":random.uniform(40,80),"max_life":80,"r":random.uniform(8,20)})
        orb_layer=get_buf(frame.shape); alive=[]
        for orb in self.orbs:
            orb["x"]+=orb["vx"]; orb["y"]+=orb["vy"]; orb["life"]-=1
            if 0<orb["x"]<w and 0<orb["y"]<h and orb["life"]>0:
                a=orb["life"]/orb["max_life"]; r=max(1,int(orb["r"]*a)); cv2.circle(orb_layer,(int(orb["x"]),int(orb["y"])),r,b_col,-1); alive.append(orb)
        self.orbs=alive; cv2.add(frame,orb_layer,dst=frame)
        self._draw_armour(frame,face_data,b_col,h,w)
        self._spawn_body_flames(face_data,h,w)
        fl_layer=get_buf(frame.shape); alive_f=[]
        for fl in self.flames:
            fl["x"]+=fl["vx"]; fl["y"]+=fl["vy"]; fl["vx"]+=random.uniform(-0.5,0.5); fl["life"]-=1
            if fl["life"]>0:
                a=fl["life"]/fl["max_life"]; sz=max(1,int(fl["size"]*a))
                col=variant_color((180,200,255) if a>0.7 else b_col if a>0.4 else tuple(int(c*a*2) for c in b_col),variant)
                px=int(fl["x"]); py=int(fl["y"])
                if 0<=px<w and 0<=py<h: cv2.circle(fl_layer,(px,py),sz,col,-1)
                alive_f.append(fl)
        self.flames=alive_f
        if np.any(fl_layer>0):
            cv2.addWeighted(cv2.GaussianBlur(fl_layer,(9,9),0),0.5,frame,1.0,0,frame)
            cv2.addWeighted(fl_layer,0.6,frame,1.0,0,frame)

class KirinEffect:
    def __init__(self): self.active=False; self.t=0; self.cx=self.cy=0; self.phase=0
    def start(self,cx,cy): self.active=True; self.t=0; self.cx=cx; self.cy=cy; self.phase=0
    def stop(self): self.active=False
    def _jagged(self,layer,x1,y1,x2,y2,depth,color,thick):
        h,w=layer.shape[:2]
        if depth==0 or math.hypot(x2-x1,y2-y1)<5: cv2.line(layer,clip_pt(x1,y1,w,h),clip_pt(x2,y2,w,h),color,thick,cv2.LINE_AA); return
        mx=(x1+x2)/2+random.uniform(-35,35); my=(y1+y2)/2+random.uniform(-35,35)
        self._jagged(layer,x1,y1,mx,my,depth-1,color,thick); self._jagged(layer,mx,my,x2,y2,depth-1,color,thick)
    def update(self,frame,cx,cy,scale=1.0,variant=0.0):
        if not self.active: return
        self.cx=cx; self.cy=cy; self.t+=1; h,w=frame.shape[:2]
        self.phase=0 if self.t<=30 else (1 if self.t<=55 else 2)
        arc_col=variant_color((200,220,255),variant)
        if self.phase==0:
            fast_glow(frame,w//2,60,200,(30,30,50),self.t/30*0.7)
            if random.random()<0.5:
                l_layer=get_buf(frame.shape)
                for _ in range(2):
                    lx1=random.randint(w//4,3*w//4); ly1=random.randint(10,80)
                    self._jagged(l_layer,lx1,ly1,lx1+random.randint(-150,150),ly1+random.randint(20,80),3,arc_col,2)
                cv2.add(frame,l_layer,dst=frame)
        elif self.phase==1:
            progress=(self.t-30)/25; white=np.full_like(frame,255)
            cv2.addWeighted(white,min(0.88,progress*1.5),frame,1.0-min(0.88,progress*1.5),0,frame)
            l_layer=get_buf(frame.shape); bolt_y=int(progress*cy)
            for width,col in [(10,(180,200,240)),(5,(220,235,255)),(2,(255,255,255))]: self._jagged(l_layer,w//2,0,cx,min(bolt_y,cy),5,col,width)
            cv2.add(frame,l_layer,dst=frame)
            if progress>0.7: fast_glow(frame,cx,cy,int(80*progress),arc_col,0.8)
        else:
            fast_glow(frame,cx,cy,int(90*scale),arc_col,0.45); fast_glow(frame,cx,cy,int(45*scale),(255,255,255),0.65)
            l_layer=get_buf(frame.shape)
            for i in range(int(12*scale)):
                angle=(i/12)*2*math.pi+self.t*0.20+random.uniform(-0.3,0.3); length=random.uniform(80,190)*scale
                ex=cx+math.cos(angle)*length; ey=cy+math.sin(angle)*length
                self._jagged(l_layer,cx,cy,ex,ey,4,arc_col,4); self._jagged(l_layer,cx,cy,ex,ey,2,(255,255,255),1)
            for _ in range(int(18*scale)):
                a=random.uniform(0,2*math.pi); d=random.uniform(15,130)*scale; sx=int(cx+math.cos(a)*d); sy=int(cy+math.sin(a)*d)
                if 0<=sx<w and 0<=sy<h: cv2.circle(l_layer,(sx,sy),random.randint(1,5),(255,255,220),-1)
            cv2.add(frame,l_layer,dst=frame)

class WaterDragonEffect:
    def __init__(self): self.active=False; self.t=0; self.particles=[]; self.cx=self.cy=0; self.wx=self.wy=0
    def start(self,cx,cy,wx=None,wy=None):
        self.active=True; self.t=0; self.particles=[]; self.cx=cx; self.cy=cy
        self.wx=wx if wx is not None else cx; self.wy=wy if wy is not None else cy+120
    def stop(self): self.active=False; self.particles=[]
    def update(self,frame,cx,cy,wrist_x=None,wrist_y=None,scale=1.0,variant=0.0):
        if not self.active: return
        self.cx=cx; self.cy=cy; self.t+=1; h,w=frame.shape[:2]
        if wrist_x is not None: self.wx=wrist_x; self.wy=wrist_y
        arm_dx=self.cx-self.wx; arm_dy=self.cy-self.wy
        arm_len=max(1,math.hypot(arm_dx,arm_dy)); ux=arm_dx/arm_len; uy=arm_dy/arm_len
        px2=-uy; py2=ux
        water_col=variant_color((255,200,100),variant)
        for i in range(12):
            t=i/11; lpx=int(self.wx+arm_dx*t); lpy=int(self.wy+arm_dy*t)
            if 0<=lpx<w and 0<=lpy<h: fast_glow(frame,lpx,lpy,int(18*scale),water_col,0.12)
        layer=get_buf(frame.shape)
        arm_extent=arm_len*1.3*scale
        for j in range(4):
            pts=[]
            for i in range(50):
                frac=i/49; along=frac*arm_extent
                ax=self.wx+ux*along; ay=self.wy+uy*along
                spiral_r=int((28+10*math.sin(frac*math.pi))*scale)
                angle=frac*4*math.pi-self.t*0.18+j*(2*math.pi/4)
                sx=ax+px2*math.cos(angle)*spiral_r+ux*math.sin(angle)*spiral_r*0.3
                sy=ay+py2*math.cos(angle)*spiral_r+uy*math.sin(angle)*spiral_r*0.3
                px3=max(0,min(w-1,int(sx))); py3=max(0,min(h-1,int(sy)))
                pts.append([px3,py3])
                if i%8==0 and len(self.particles)<350:
                    self.particles.append({"x":float(sx),"y":float(sy),"vx":random.uniform(-2,2)+(px2*math.cos(angle))*2,"vy":random.uniform(-3,-0.5)+(py2*math.cos(angle))*2,"life":random.uniform(15,40),"max_life":40,"size":random.uniform(2,7)*scale})
            if len(pts)>1:
                for i in range(len(pts)-1):
                    frac=i/49; t2=abs(math.sin(frac*math.pi)); thick=max(1,int((8+t2*8)*scale))
                    cv2.line(layer,tuple(pts[i]),tuple(pts[i+1]),water_col,thick,cv2.LINE_AA)
                    cv2.line(layer,tuple(pts[i]),tuple(pts[i+1]),(255,255,255),max(1,thick//3),cv2.LINE_AA)
        small=cv2.resize(layer,(max(1,w//4),max(1,h//4))); small=cv2.GaussianBlur(small,(5,5),0)
        cv2.addWeighted(cv2.resize(small,(w,h)),0.5,frame,1.0,0,frame); cv2.add(frame,layer,dst=frame)
        p_layer=get_buf(frame.shape); alive=[]
        for p in self.particles:
            p["x"]+=p["vx"]; p["y"]+=p["vy"]; p["vy"]-=0.10; p["life"]-=1
            if p["life"]>0:
                a=p["life"]/p["max_life"]; sz=max(1,int(p["size"]*a)); col=variant_color((int(255*a),int(220*a),int(120*a)),variant)
                px4=int(p["x"]); py4=int(p["y"])
                if 0<=px4<w and 0<=py4<h: cv2.circle(p_layer,(px4,py4),sz,col,-1)
                alive.append(p)
        self.particles=alive; cv2.add(frame,p_layer,dst=frame)
        tip_x=max(0,min(w-1,int(self.cx+ux*40*scale))); tip_y=max(0,min(h-1,int(self.cy+uy*40*scale)-int(30*scale)))
        fast_glow(frame,tip_x,tip_y,int(30*scale),water_col,0.5)
        head_layer=get_buf(frame.shape)
        cv2.circle(head_layer,(tip_x,tip_y),int(18*scale),variant_color((200,210,255),variant),-1)
        cv2.circle(head_layer,(tip_x,tip_y),int(18*scale),(255,255,255),2,cv2.LINE_AA)
        for eside in [-1,1]:
            ex2=max(0,min(w-1,tip_x+eside*int(7*scale))); ey2=max(0,min(h-1,tip_y-int(5*scale)))
            cv2.circle(head_layer,(ex2,ey2),max(1,int(3*scale)),(0,200,255),-1)
        cv2.add(frame,head_layer,dst=frame)

class AmaterasuEffect:
    def __init__(self): self.active=False; self.t=0; self.particles=[]
    def start(self): self.active=True; self.t=0; self.particles=[]
    def stop(self): self.active=False; self.particles=[]
    def update(self,frame,face_data,scale=1.0,variant=0.0):
        if not self.active or face_data is None: return
        self.t+=1; h,w=frame.shape[:2]
        for eye_center,eye_r in [(face_data["left_eye"],face_data["left_r"]),(face_data["right_eye"],face_data["right_r"])]:
            ex,ey=eye_center; r=max(7,int(eye_r*1.05)); fast_glow(frame,ex,ey,int(r*3),(0,0,80),0.5)
            if len(self.particles)<400:
                for _ in range(10):
                    sa=random.uniform(0,2*math.pi); sp=random.uniform(2,8)*scale
                    self.particles.append({"x":float(ex+math.cos(sa)*r),"y":float(ey+math.sin(sa)*r),"vx":math.cos(sa)*sp,"vy":math.sin(sa)*sp-random.uniform(1,3),"life":random.uniform(25,50),"max_life":50,"size":random.uniform(4,14)*scale,"layer":random.randint(0,2)})
        COLS=[[(30,30,180),(20,20,120)],[(10,10,80),(5,5,50)],[(0,0,0),(0,0,0)]]; p_layer=get_buf(frame.shape); alive=[]
        for p in self.particles:
            p["x"]+=p["vx"]; p["y"]+=p["vy"]; p["vy"]-=0.12; p["vx"]+=random.uniform(-0.3,0.3); p["life"]-=1
            if p["life"]>0:
                a=p["life"]/p["max_life"]; sz=max(1,int(p["size"]*a)); cols=COLS[p["layer"]]; color=cols[0] if a>0.5 else cols[1]
                px=int(p["x"]); py=int(p["y"])
                if 0<=px<w and 0<=py<h: cv2.circle(p_layer,(px,py),sz,color,-1)
                alive.append(p)
        self.particles=alive; cv2.addWeighted(frame,0.88,np.zeros_like(frame),0.12,0,frame); cv2.add(frame,p_layer,dst=frame)

class SandShieldEffect:
    def __init__(self): self.active=False; self.t=0; self.particles=[]; self.streams=[]; self.cx=self.cy=0
    def start(self,cx,cy): self.active=True; self.t=0; self.particles=[]; self.streams=[]; self.cx=cx; self.cy=cy
    def stop(self): self.active=False; self.particles=[]; self.streams=[]
    def _spawn_ring(self,cx,cy,ring,scale):
        r_base=(60+ring*55)*scale; n=18+ring*12
        for i in range(n):
            a=(i/n)*2*math.pi+self.t*0.03*(1+ring*0.35)*(-1 if ring%2 else 1)+random.uniform(-0.08,0.08)
            r=r_base+random.uniform(-20,20)*scale
            self.particles.append({"x":cx+math.cos(a)*r,"y":cy+math.sin(a)*r,"vx":(-math.sin(a))*(1.8+ring*0.6)*scale+random.uniform(-0.5,0.5),"vy":(math.cos(a))*(1.8+ring*0.6)*scale+random.uniform(-0.5,0.5)-0.4,"life":random.uniform(8,22),"max_life":22,"size":random.uniform(1.5,4+ring*2)*scale,"ring":ring,"shape":random.choice(["dot","grain"])})
    def _spawn_stream(self,cx,cy,scale):
        for _ in range(2):
            ax=cx+random.uniform(-80,80)*scale; ay=cy+random.uniform(60,140)*scale
            self.streams.append({"x":float(ax),"y":float(ay),"vx":random.uniform(-1.5,1.5),"vy":-random.uniform(4,9)*scale,"life":random.uniform(20,40),"max_life":40,"size":random.uniform(2,6)*scale})
    def update(self,frame,cx,cy,scale=1.0,variant=0.0):
        if not self.active: return
        self.cx=cx; self.cy=cy; self.t+=1; h,w=frame.shape[:2]
        for ring in range(5): self._spawn_ring(cx,cy,ring,scale)
        self._spawn_stream(cx,cy,scale)
        fast_glow(frame,cx,cy,int(130*scale),(20,80,120),0.28); fast_glow(frame,cx,cy,int(70*scale),(30,110,160),0.32)
        floor_y=cy+int(80*scale); floor_layer=get_buf(frame.shape)
        for _ in range(60):
            a=random.uniform(0,2*math.pi)+self.t*0.04; rd=random.uniform(30,90)*scale
            fx=max(0,min(w-1,int(cx+math.cos(a)*rd))); fy=max(0,min(h-1,int(floor_y+math.sin(a)*rd*0.3)))
            cv2.circle(floor_layer,(fx,fy),max(1,int(random.uniform(1,4)*scale)),(20,90,130),-1)
        cv2.addWeighted(cv2.GaussianBlur(floor_layer,(7,7),0),0.4,frame,1.0,0,frame)
        RING_COLS=[(40,140,200),(30,115,175),(22,95,155),(15,75,135),(8,55,110)]
        p_layer=get_buf(frame.shape); alive=[]
        for p in self.particles:
            p["x"]+=p["vx"]; p["y"]+=p["vy"]; p["life"]-=1
            if p["life"]>0:
                a=p["life"]/p["max_life"]; ring=p["ring"]; sz=max(1,int(p["size"]*a))
                base=RING_COLS[min(ring,4)]; color=variant_color(tuple(min(255,int(c*(a*(0.6+random.random()*0.5)))) for c in base),variant*0.3)
                px=int(p["x"]); py=int(p["y"])
                if 0<=px<w and 0<=py<h:
                    if p["shape"]=="grain":
                        angle2=math.atan2(p["vy"],p["vx"]); ex2=max(0,min(w-1,int(px+math.cos(angle2)*sz*2))); ey2=max(0,min(h-1,int(py+math.sin(angle2)*sz*2)))
                        cv2.line(p_layer,(px,py),(ex2,ey2),color,max(1,sz//2))
                    else: cv2.circle(p_layer,(px,py),sz,color,-1)
                alive.append(p)
        self.particles=alive
        s_layer=get_buf(frame.shape); alive_s=[]
        for s in self.streams:
            s["x"]+=s["vx"]; s["y"]+=s["vy"]; s["vx"]+=random.uniform(-0.3,0.3); s["life"]-=1
            if s["life"]>0:
                a=s["life"]/s["max_life"]; sz=max(1,int(s["size"]*a)); color=variant_color((int(40*a),int(130*a),int(185*a)),variant*0.3)
                px=int(s["x"]); py=int(s["y"])
                if 0<=px<w and 0<=py<h: cv2.circle(s_layer,(px,py),sz,color,-1)
                alive_s.append(s)
        self.streams=alive_s
        cv2.addWeighted(cv2.GaussianBlur(p_layer,(3,3),0),0.35,frame,1.0,0,frame)
        cv2.add(frame,p_layer,dst=frame); cv2.add(frame,s_layer,dst=frame)

# ═══════════════════════════════════════════════════════════════
#  MAIN CV LOOP
# ═══════════════════════════════════════════════════════════════
def cv_loop():
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_FPS,60); cap.set(cv2.CAP_PROP_AUTOFOCUS,1)

    hand_thread=DetectorThread(hand_options,HandLandmarker)
    face_thread=DetectorThread(face_options,FaceLandmarker)
    cnn_thread=CNNThread()
    hand_thread.start(); face_thread.start(); cnn_thread.start()

    engine=SequenceEngine(JUTSU_SEQUENCES); smoother=PredictionSmoother(); screen=ScreenEffects()
    fireball=FireballEffect(); chidori=ChidoriEffect(); rasengan=RasenganEffect()
    shadow_clone=ShadowCloneEffect(); sharingan=SharinganEffect()
    flamethrower=FlamethrowerEffect(); susanoo=SusanooEffect()
    kirin=KirinEffect(); water_dragon=WaterDragonEffect()
    amaterasu=AmaterasuEffect(); sand_shield=SandShieldEffect()
    all_effects=[fireball,chidori,rasengan,shadow_clone,flamethrower,susanoo,
                 kirin,water_dragon,amaterasu,sand_shield]

    sharingan_active=False; active_jutsu=None; current_sign=None; current_conf=0.0
    last_cx,last_cy=640,360; last_wrist_x,last_wrist_y=640,480
    fps=0.0; fps_timer=time.time(); fps_count=0
    control_scale=1.0; control_variant=0.0

    anchor = HandAnchor()   # identity-anchor for stable hand tracking

    blank=np.zeros((720,1280,3),np.uint8)
    hand_thread.push_frame(blank); face_thread.push_frame(blank); time.sleep(0.3)
    print("CV loop started — open http://127.0.0.1:5000")

    while True:
        # ── Reset ──────────────────────────────────────────────
        with state_lock:
            if STATE.get("_reset"):
                for fx in all_effects: fx.stop()
                engine.reset(); smoother.reset()
                active_jutsu=None; current_sign=None; current_conf=0.0
                control_scale=1.0; control_variant=0.0
                anchor.unlock()
                STATE["_reset"]=False

        ret,frame=cap.read()
        if not ret: continue
        frame=cv2.flip(frame,1); h,w=frame.shape[:2]

        hand_thread.push_frame(frame); face_thread.push_frame(frame)
        hand_result=hand_thread.get_result(); face_result=face_thread.get_result()
        face_data=get_face_data(face_result,w,h)
        bbox=get_both_hands_bbox(hand_result,w,h)

        # ── Resolve hands by wrist-proximity identity matching ──
        # effect_pts  = landmarks of the effect hand  (or None)
        # control_pts = landmarks of the other hand   (or None)
        effect_pts, control_pts = anchor.resolve(hand_result, w, h)

        # ── Read control hand scale + hue ───────────────────────
        if active_jutsu is not None and control_pts is not None:
            control_scale   = get_pinch_scale(control_pts)
            control_variant = get_wrist_roll(control_pts)

        # ── Update effect hand position ─────────────────────────
        if active_jutsu in ("fireball","chidori","rasengan","kirin",
                            "water_dragon","sand_shield","shadow_clone"):
            if effect_pts is not None:
                last_cx, last_cy       = pts_center(effect_pts)
                last_wrist_x, last_wrist_y = effect_pts[0]

        # ── Sign detection (only when no jutsu active) ──────────
        if active_jutsu is None:
            if bbox:
                cnn_thread.push_data(frame, bbox)
                sign, conf = cnn_thread.get_result()
                smoothed   = smoother.update(sign, conf)
                current_sign = smoothed; current_conf = conf if smoothed else 0.0

                # Position from all visible hands while signing
                if hand_result and hand_result.hand_landmarks:
                    all_pts = [(int(lm.x*w), int(lm.y*h))
                               for hand in hand_result.hand_landmarks for lm in hand]
                    last_cx = int(np.mean([p[0] for p in all_pts]))
                    last_cy = int(np.mean([p[1] for p in all_pts]))
                    last_wrist_x = int(hand_result.hand_landmarks[0][0].x * w)
                    last_wrist_y = int(hand_result.hand_landmarks[0][0].y * h)

                triggered = engine.update(smoothed) if smoothed else None

                if triggered:
                    final_jutsu = triggered
                    if triggered in COMBOS:
                        req, replacement = COMBOS[triggered]
                        if req == "sharingan" and sharingan_active:
                            final_jutsu = replacement

                    # Lock effect hand: whichever wrist is closest to
                    # the last signing position becomes the effect hand.
                    # The anchor stores that wrist pixel and uses it for
                    # identity matching on every subsequent frame.
                    if hand_result and hand_result.hand_landmarks:
                        best_d = float('inf')
                        best_wrist = (last_wrist_x, last_wrist_y)
                        for hand in hand_result.hand_landmarks:
                            wx = int(hand[0].x * w)
                            wy = int(hand[0].y * h)
                            d  = math.hypot(wx - last_cx, wy - last_cy)
                            if d < best_d:
                                best_d = d; best_wrist = (wx, wy)
                        anchor.lock(best_wrist)
                    else:
                        anchor.lock((last_wrist_x, last_wrist_y))

                    play_sound(final_jutsu)
                    col = JUTSU_COLORS_BGR.get(final_jutsu, (255,255,255))
                    flash_col = (255,255,255)
                    if final_jutsu=="kirin":          flash_col=(230,240,255)
                    elif final_jutsu=="amaterasu":    flash_col=(20,20,20)
                    elif final_jutsu=="shadow_clone": flash_col=(180,200,220)
                    screen.trigger(col, flash_color=flash_col,
                                   shake=10 if final_jutsu=="shadow_clone" else 14)

                    if sharingan_active and final_jutsu != "sharingan":
                        upgraded = sharingan.on_jutsu_cast()
                        if upgraded:
                            play_sound("mangekyou")
                            screen.trigger(JUTSU_COLORS_BGR["mangekyou"],
                                           flash_color=(100,0,150), shake=20)

                    if final_jutsu=="sharingan":
                        sharingan_active = not sharingan_active
                        if sharingan_active: sharingan.start()
                        else: sharingan.stop(); sharingan.reset_mangekyou()
                    elif final_jutsu=="susanoo":
                        if sharingan_active: active_jutsu="susanoo"; susanoo.start()
                    elif final_jutsu=="amaterasu":
                        if sharingan_active: active_jutsu="amaterasu"; amaterasu.start()
                    elif final_jutsu=="kirin":        active_jutsu="kirin";        kirin.start(last_cx,last_cy)
                    elif final_jutsu=="fireball":     active_jutsu="fireball";     fireball.start(last_cx,last_cy)
                    elif final_jutsu=="chidori":      active_jutsu="chidori";      chidori.start(last_cx,last_cy)
                    elif final_jutsu=="rasengan":     active_jutsu="rasengan";     rasengan.start(last_cx,last_cy)
                    elif final_jutsu=="shadow_clone": active_jutsu="shadow_clone"; shadow_clone.start(frame)
                    elif final_jutsu=="flamethrower": active_jutsu="flamethrower"; flamethrower.start()
                    elif final_jutsu=="water_dragon": active_jutsu="water_dragon"; water_dragon.start(last_cx,last_cy,last_wrist_x,last_wrist_y)
                    elif final_jutsu=="sand_shield":  active_jutsu="sand_shield";  sand_shield.start(last_cx,last_cy)
            else:
                current_sign=None; current_conf=0.0

        # ── Bounding box while idle ──────────────────────────────
        if active_jutsu is None and bbox:
            x1,y1,x2,y2=bbox; cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,0),1)

        # ── Render effects ───────────────────────────────────────
        sharingan.update(frame, face_data)
        if   active_jutsu=="fireball":     fireball.update(frame,last_cx,last_cy,control_scale,control_variant)
        elif active_jutsu=="chidori":      chidori.update(frame,last_cx,last_cy,control_scale,control_variant)
        elif active_jutsu=="rasengan":     rasengan.update(frame,last_cx,last_cy,control_scale,control_variant)
        elif active_jutsu=="shadow_clone": shadow_clone.update(frame,control_scale,control_variant)
        elif active_jutsu=="flamethrower": flamethrower.update(frame,face_data,control_scale,control_variant)
        elif active_jutsu=="susanoo":      susanoo.update(frame,face_data,control_scale,control_variant)
        elif active_jutsu=="kirin":        kirin.update(frame,last_cx,last_cy,control_scale,control_variant)
        elif active_jutsu=="water_dragon": water_dragon.update(frame,last_cx,last_cy,last_wrist_x,last_wrist_y,control_scale,control_variant)
        elif active_jutsu=="amaterasu":    amaterasu.update(frame,face_data,control_scale,control_variant)
        elif active_jutsu=="sand_shield":  sand_shield.update(frame,last_cx,last_cy,control_scale,control_variant)

        frame = screen.apply(frame)

        # ── Control hand HUD (ring + readout on control hand) ───
        if active_jutsu is not None and control_pts is not None:
            ctrl_cx, ctrl_cy = pts_center(control_pts)
            ring_r = max(10, int(30 * control_scale))
            cv2.circle(frame, (ctrl_cx, ctrl_cy), ring_r, (0,220,255), 1, cv2.LINE_AA)
            cv2.putText(frame,
                        f"scale:{control_scale:.1f}  hue:{control_variant:.2f}",
                        (ctrl_cx - 45, ctrl_cy - ring_r - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,220,255), 1)

        fps_count += 1
        if time.time()-fps_timer >= 0.5:
            fps = fps_count/(time.time()-fps_timer); fps_count=0; fps_timer=time.time()

        with state_lock:
            STATE["current_sign"]    = current_sign
            STATE["conf"]            = round(current_conf, 3)
            STATE["progress"]        = engine.get_progress() if active_jutsu is None else []
            STATE["active_jutsu"]    = active_jutsu
            STATE["sharingan_active"]= sharingan_active
            STATE["mangekyou"]       = sharingan.mangekyou
            STATE["fps"]             = round(fps, 1)
            STATE["jutsu_count"]     = sharingan.jutsu_count
            STATE["control_scale"]   = round(control_scale, 2)
            STATE["control_variant"] = round(control_variant, 2)

        ok,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,82])
        if ok: _frame_queue.append(buf.tobytes())

# ═══════════════════════════════════════════════════════════════
#  FLASK
# ═══════════════════════════════════════════════════════════════
@app.route("/")
def index():
    jutsu_data={k:{"name":JUTSU_NAMES[k],"seq":JUTSU_SEQUENCES.get(k,[])}
                for k in list(JUTSU_SEQUENCES.keys())+["kirin","mangekyou"]}
    return render_template("index.html",jutsu_data=jutsu_data,jutsu_names=JUTSU_NAMES)

def _generate():
    while True:
        if _frame_queue:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+_frame_queue[-1]+b"\r\n")
        else: time.sleep(0.01)

@app.route("/video_feed")
def video_feed():
    return Response(_generate(),mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/state")
def api_state():
    with state_lock: return jsonify(dict(STATE))

@app.route("/api/reset",methods=["POST"])
def api_reset():
    with state_lock: STATE["_reset"]=True
    return jsonify({"ok":True})

if __name__=="__main__":
    os.makedirs("templates",exist_ok=True)
    t=threading.Thread(target=cv_loop,daemon=True); t.start()
    time.sleep(1.5)
    app.run(host="0.0.0.0",port=5000,debug=False,threaded=True)