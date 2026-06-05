

import sys, os, urllib.request, subprocess, collections

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args, "--quiet"])
for mod, pkg in [("cv2","opencv-python"), ("mediapipe","mediapipe")]:
    try: __import__(mod)
    except ImportError: pip(pkg)

import cv2, mediapipe as mp, numpy as np, math, time

# ── Model download ────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading model (~6 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.\n")

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)
print("[OK] ULTRA ROBOT HAND READY\n")

# ── Colors ────────────────────────────────────────────────────────────────────
W    = (255, 255, 255)
W80  = (200, 200, 200)
W40  = (100, 100, 100)
OR   = (30,  70,  255)   # orange-red
OR2  = (10,  40,  180)   # darker orange
BLU  = (200, 110,  20)   # blue grid
CYAN = (255, 220,  80)   # accent
DIM  = (80,  80,  80)

CHAINS   = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
PALM_IDS = [0,1,5,9,13,17]
TIPS     = [4,8,12,16,20]
KNUCKLES = [1,5,9,13,17]
MID      = [2,3,6,7,10,11,14,15,18,19]

# ── Inertia / lag state ───────────────────────────────────────────────────────
lag_ring_angle   = 0.0   # lagging outer ring angle
lag_ring_vel     = 0.0
lag_inner_angle  = 0.0
lag_inner_vel    = 0.0
prev_palm_angle  = None
palm_angle_smooth= 0.0

# tip trails
TIP_TRAIL_LEN = 12
tip_trails = {i: collections.deque(maxlen=TIP_TRAIL_LEN) for i in TIPS}

# smoothed landmarks for lag
smooth_pts   = None
SMOOTH_ALPHA = 0.35   # lower = more lag/smoothing

# arm smoothing
smooth_wrist = None
ARM_ALPHA    = 0.25


# ════════════════════════════════════════════════════════════════════════════
#  DRAWING PRIMITIVES
# ════════════════════════════════════════════════════════════════════════════

def lerp(a, b, t):
    return a + (b - a) * t

def lerp_pt(a, b, t):
    return (int(a[0]+(b[0]-a[0])*t), int(a[1]+(b[1]-a[1])*t))

def dist2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def draw_glowing_line(img, p1, p2, color, thickness=1, alpha=1.0):
    """Line with a soft bloom — draw thick dim then thin bright"""
    if alpha <= 0: return
    c_dim = tuple(int(c * 0.3 * alpha) for c in color)
    c_bright = tuple(int(c * alpha) for c in color)
    cv2.line(img, p1, p2, c_dim,    thickness+4, cv2.LINE_AA)
    cv2.line(img, p1, p2, c_bright, thickness,   cv2.LINE_AA)


def draw_mech_joint(img, cx, cy, size=16, t=0.0, depth=1.0):
    """
    Full mechanical joint:
    - outer ring with tick spokes
    - spinning inner arc
    - second inner ring
    - center filled dot
    - optional depth fade
    """
    alpha = max(0.3, depth)
    R_OUT  = int(size)
    R_MID  = int(size * 0.7)
    R_IN   = int(size * 0.42)
    R_DOT  = max(2, int(size * 0.22))

    c = tuple(int(x * alpha) for x in W)

    # outer ring
    cv2.circle(img, (cx,cy), R_OUT, c, 1, cv2.LINE_AA)
    # mid ring
    cv2.circle(img, (cx,cy), R_MID, c, 1, cv2.LINE_AA)
    # inner ring
    cv2.circle(img, (cx,cy), R_IN,  c, 1, cv2.LINE_AA)

    # spinning arc on mid ring
    arc_start = int((t * 140 + cx) % 360)
    cv2.ellipse(img, (cx,cy), (R_MID, R_MID), arc_start, 10, 80,
                tuple(int(x*min(1.0,alpha*1.2)) for x in W), 2, cv2.LINE_AA)

    # 4 crosshair spokes
    for deg in (0, 90, 180, 270):
        a  = math.radians(deg + t*20)   # slowly rotate spokes
        x1 = int(cx + (R_OUT-2)*math.cos(a)); y1 = int(cy + (R_OUT-2)*math.sin(a))
        x2 = int(cx + (R_OUT+6)*math.cos(a)); y2 = int(cy + (R_OUT+6)*math.sin(a))
        cv2.line(img, (x1,y1),(x2,y2), c, 1, cv2.LINE_AA)

    # center dot
    cv2.circle(img, (cx,cy), R_DOT, c, -1, cv2.LINE_AA)


def draw_small_joint(img, cx, cy, size=7, t=0.0, depth=1.0):
    alpha = max(0.3, depth)
    c = tuple(int(x*alpha) for x in W)
    cv2.circle(img, (cx,cy), size,       c, 1, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), max(2,size-3), c, 1, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), 2,          c, -1, cv2.LINE_AA)

    arc_s = int((t*180+cy)%360)
    cv2.ellipse(img,(cx,cy),(size,size),arc_s,0,60,c,1,cv2.LINE_AA)


def draw_palm_reactor(img, cx, cy, t, lag_a, lag_b, palm_vel):
    """
    The big reactor circle on the palm.
    - 6 concentric rings
    - 2 lagging ring sets (outer lags behind motion)
    - spinning arc pairs (clockwise + counter)
    - radial ticks
    - inner asterisk
    - pulsing center based on hand velocity
    """
    pulse = 1.0 + 0.12 * math.sin(t * 4.0)   # breathing pulse
    BASE  = int(52 * pulse)

    rings = [BASE, int(BASE*0.84), int(BASE*0.68), int(BASE*0.52), int(BASE*0.36), int(BASE*0.20)]

    # draw rings with slight glow
    for i, r in enumerate(rings):
        alpha = 0.7 + 0.3*(i/len(rings))
        c = tuple(int(x*alpha) for x in W)
        cv2.circle(img, (cx,cy), r, c, 1, cv2.LINE_AA)

    # ── Lagging outer arc (follows hand rotation with inertia) ──
    cv2.ellipse(img, (cx,cy), (rings[0]+8, rings[0]+8),
                int(lag_a) % 360, 0, 75, W, 2, cv2.LINE_AA)
    cv2.ellipse(img, (cx,cy), (rings[0]+8, rings[0]+8),
                (int(lag_a)+180) % 360, 0, 75, W, 2, cv2.LINE_AA)

    # ── Secondary lagging ring ──
    cv2.ellipse(img, (cx,cy), (rings[1]+5, rings[1]+5),
                int(lag_b) % 360, 0, 50, W80, 1, cv2.LINE_AA)
    cv2.ellipse(img, (cx,cy), (rings[1]+5, rings[1]+5),
                (int(lag_b)+180) % 360, 0, 50, W80, 1, cv2.LINE_AA)

    # ── Fast spinning arcs on ring[2] ──
    fa = int((t * 160) % 360)
    cv2.ellipse(img,(cx,cy),(rings[2],rings[2]), fa,    0, 60, W, 2, cv2.LINE_AA)
    cv2.ellipse(img,(cx,cy),(rings[2],rings[2]), fa+180,0, 60, W, 2, cv2.LINE_AA)

    # ── Slow counter-spin on ring[3] ──
    fb = int((360 - t*90) % 360)
    cv2.ellipse(img,(cx,cy),(rings[3],rings[3]), fb,   0, 45, W80, 1, cv2.LINE_AA)

    # ── 12 radial tick marks ──
    for i in range(12):
        a = math.radians(i*30 + t*18)
        cv2.line(img,
                 (int(cx + rings[0]*math.cos(a)),     int(cy + rings[0]*math.sin(a))),
                 (int(cx + (rings[0]+10)*math.cos(a)), int(cy + (rings[0]+10)*math.sin(a))),
                 W, 1, cv2.LINE_AA)

    # ── Inner asterisk (8-point star) ──
    for a_deg in range(0, 180, 22):
        a = math.radians(a_deg + t*15)
        r_in = rings[4]
        cv2.line(img,
                 (int(cx - r_in*math.cos(a)), int(cy - r_in*math.sin(a))),
                 (int(cx + r_in*math.cos(a)), int(cy + r_in*math.sin(a))),
                 W, 1, cv2.LINE_AA)

    # ── Pulsing center dot ──
    pulse_r = int(6 + 3*math.sin(t*6))
    cv2.circle(img, (cx,cy), pulse_r, W, -1, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), pulse_r+4, W40, 1, cv2.LINE_AA)


def draw_arm(img, wrist_pt, frame_w, frame_h, t):
    """
    Estimate elbow + shoulder from wrist position and draw
    a robotic arm segment with dual-rail lines + joint circles.
    """
    wx, wy = wrist_pt

    # Estimate elbow: below wrist (toward bottom of frame), slightly offset
    elbow_x = int(wx + (frame_w*0.5 - wx)*0.18)
    elbow_y = min(frame_h - 20, wy + int(frame_h * 0.22))

    # Estimate shoulder: further below, center of frame
    shoulder_x = int(frame_w * 0.5)
    shoulder_y = min(frame_h - 10, wy + int(frame_h * 0.42))

    # ── Draw forearm (wrist → elbow) ──
    _draw_arm_segment(img, (wx,wy), (elbow_x, elbow_y), width=14, t=t, seg_id=0)

    # ── Draw upper arm (elbow → shoulder) ──
    _draw_arm_segment(img, (elbow_x,elbow_y), (shoulder_x, shoulder_y), width=18, t=t, seg_id=1)

    # Elbow joint
    _draw_arm_joint(img, elbow_x, elbow_y, r=20, t=t)

    # Shoulder joint (partial, near edge of frame)
    if shoulder_y < frame_h - 15:
        _draw_arm_joint(img, shoulder_x, shoulder_y, r=26, t=t)


def _draw_arm_segment(img, p1, p2, width=14, t=0.0, seg_id=0):
    """Dual-rail robotic arm segment with cross-struts"""
    dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
    length = max(math.hypot(dx,dy), 1)
    nx = -dy/length; ny = dx/length   # normal

    hw = width//2
    # Four corner points
    l1 = (int(p1[0]+nx*hw), int(p1[1]+ny*hw))
    l2 = (int(p2[0]+nx*hw), int(p2[1]+ny*hw))
    r1 = (int(p1[0]-nx*hw), int(p1[1]-ny*hw))
    r2 = (int(p2[0]-nx*hw), int(p2[1]-ny*hw))

    # outer rails
    cv2.line(img, l1, l2, W80, 1, cv2.LINE_AA)
    cv2.line(img, r1, r2, W80, 1, cv2.LINE_AA)

    # center line (thinner)
    cv2.line(img, p1, p2, W40, 1, cv2.LINE_AA)

    # cross-struts along the segment
    n_struts = max(2, int(length / 28))
    for i in range(1, n_struts):
        frac = i / n_struts
        mx = int(p1[0]+dx*frac); my = int(p1[1]+dy*frac)
        s1 = (int(mx+nx*hw), int(my+ny*hw))
        s2 = (int(mx-nx*hw), int(my-ny*hw))
        cv2.line(img, s1, s2, W40, 1, cv2.LINE_AA)

    # rivet dots on rails
    for i in range(0, n_struts+1):
        frac = i / max(n_struts, 1)
        for side in (1,-1):
            rx = int(p1[0]+dx*frac + nx*hw*side)
            ry = int(p1[1]+dy*frac + ny*hw*side)
            cv2.circle(img, (rx,ry), 2, W80, -1, cv2.LINE_AA)


def _draw_arm_joint(img, cx, cy, r=20, t=0.0):
    """Large mechanical elbow/shoulder joint"""
    cv2.circle(img, (cx,cy), r,       W80, 1, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), r-5,     W40, 1, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), 5,       W80, -1, cv2.LINE_AA)

    arc_s = int((t*80+cx)%360)
    cv2.ellipse(img,(cx,cy),(r,r), arc_s, 0, 90, W80, 2, cv2.LINE_AA)

    for deg in (0,90,180,270):
        a = math.radians(deg)
        cv2.line(img,
                 (int(cx+(r-3)*math.cos(a)), int(cy+(r-3)*math.sin(a))),
                 (int(cx+(r+7)*math.cos(a)), int(cy+(r+7)*math.sin(a))),
                 W80, 1, cv2.LINE_AA)


def draw_finger_flex_arc(img, base_pt, tip_pt, depth=1.0):
    """Small arc showing flex/curl of each finger"""
    bx,by = base_pt; tx,ty = tip_pt
    dx = tx-bx; dy = ty-by
    length = max(math.hypot(dx,dy),1)
    # arc radius = 1/3 finger length
    r = int(length * 0.33)
    if r < 5: return
    # angle of finger
    angle = math.degrees(math.atan2(dy,dx))
    alpha = max(0.2, depth*0.6)
    c = tuple(int(x*alpha) for x in W40)
    cv2.ellipse(img, base_pt, (r,r), angle, -30, 30, c, 1, cv2.LINE_AA)


def draw_tip_trail(img, trail, color=W):
    """Motion trail behind fingertip"""
    pts = list(trail)
    for i in range(1, len(pts)):
        alpha = i/len(pts)
        c = tuple(int(x*alpha*0.6) for x in color)
        thickness = max(1, int(alpha*3))
        cv2.line(img, pts[i-1], pts[i], c, thickness, cv2.LINE_AA)


def draw_depth_indicator(img, pts, cx, cy, t):
    """
    Show Z-depth heatmap dots on each joint:
    nearer = brighter / larger
    """
    # We fake depth from palm z coords
    pass  # handled by depth param in draw_mech_joint


def draw_skeleton(img, pts, raw_lms, t, lag_a, lag_b, frame_w, frame_h):
    # ── Arm first (behind hand) ──
    draw_arm(img, pts[0], frame_w, frame_h, t)

    # ── Bones ──
    for chain in CHAINS:
        for i in range(len(chain)-1):
            p1,p2 = pts[chain[i]], pts[chain[i+1]]
            depth = 1.0 - abs(raw_lms[chain[i]].z) * 2
            depth = max(0.35, min(1.0, depth))
            cv2.line(img, p1, p2, tuple(int(x*depth) for x in W), 1, cv2.LINE_AA)

    # palm web
    for a,b in [(1,5),(5,9),(9,13),(13,17),(0,1),(0,17)]:
        cv2.line(img, pts[a], pts[b], W40, 1, cv2.LINE_AA)

    # ── Palm reactor ──
    pcx = int(np.mean([pts[i][0] for i in PALM_IDS]))
    pcy = int(np.mean([pts[i][1] for i in PALM_IDS]))
    palm_vel = 0.0  # could add velocity
    draw_palm_reactor(img, pcx, pcy, t, lag_a, lag_b, palm_vel)

    # ── Tip trails ──
    for tip_id in TIPS:
        tip_trails[tip_id].append(pts[tip_id])
        draw_tip_trail(img, tip_trails[tip_id])

    # ── Finger flex arcs ──
    finger_bases = [1, 5, 9, 13, 17]
    finger_tips_list = [4, 8, 12, 16, 20]
    for base_id, tip_id in zip(finger_bases, finger_tips_list):
        depth = 1.0 - abs(raw_lms[base_id].z)*2
        draw_finger_flex_arc(img, pts[base_id], pts[tip_id], depth)

    # ── Joints (on top) ──
    for i, (x, y) in enumerate(pts):
        depth = 1.0 - abs(raw_lms[i].z) * 2.5
        depth = max(0.3, min(1.0, depth))
        sz_scale = 0.7 + 0.3 * depth

        if i in TIPS:
            draw_mech_joint(img, x, y, size=int(15*sz_scale), t=t+i*0.3, depth=depth)
        elif i in KNUCKLES:
            draw_mech_joint(img, x, y, size=int(12*sz_scale), t=t+i*0.2, depth=depth)
        elif i in MID:
            draw_small_joint(img, x, y, size=int(7*sz_scale), t=t+i*0.15, depth=depth)


def get_rotation(pts):
    dx = pts[9][0]-pts[0][0]; dy = pts[9][1]-pts[0][1]
    return int(math.degrees(math.atan2(-dy,dx))%360)

def get_palm_data(pts):
    wrist = pts[0]
    dists = [math.dist(pts[i],wrist) for i in TIPS]
    base  = max(math.dist(pts[0],pts[9]),1)
    return float(np.clip(np.mean(dists)/(base*2.5)*100,0,100))


def draw_cube(img, ox, oy, t):
    rad    = math.radians((t*38)%360)
    ca,sa  = math.cos(rad),math.sin(rad)
    s      = 28
    def iso(x3,y3,z3):
        rx=x3*ca-z3*sa; rz=x3*sa+z3*ca
        return (ox+int((rx-rz)*s*0.55), oy+int(-y3*s+(rx+rz)*s*0.28))
    v=[iso(-1,-1,-1),iso(1,-1,-1),iso(1,1,-1),iso(-1,1,-1),
       iso(-1,-1,1), iso(1,-1,1), iso(1,1,1), iso(-1,1,1)]
    for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        cv2.line(img,v[a],v[b],OR,1,cv2.LINE_AA)

def draw_grid(img, ox, oy):
    sp=20; cols=5; rows=3
    for r in range(rows+1):
        y=oy+r*(sp//2)
        cv2.line(img,(ox,y),(ox+cols*sp,y),BLU,1,cv2.LINE_AA)
    for c in range(cols+1):
        x=ox+c*sp
        cv2.line(img,(x,oy),(x,oy+rows*(sp//2)),BLU,1,cv2.LINE_AA)


def draw_scanline(img, t, h, w):
    """Subtle horizontal scan line sweeping down"""
    y = int((t*120)%h)
    cv2.line(img, (0,y), (w,y), (255,255,255), 1)
    # alpha blend — just draw at low opacity using addWeighted trick
    scan = np.zeros_like(img)
    scan[max(0,y-1):y+2, :] = (30,30,30)
    cv2.addWeighted(img, 1.0, scan, 0.4, 0, img)


def draw_hud(img, rot, pdat, wx, wy, t, frame_w, frame_h):
    tx = max(10, wx - 40)
    ty = min(frame_h - 100, wy + 55)

    # glitch flicker on rotation number
    glitch = math.sin(t*23)*math.sin(t*7) > 0.85
    rot_disp = rot + (np.random.randint(-3,4) if glitch else 0)

    cv2.putText(img, f"^  rotation  {rot_disp}", (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, W, 1, cv2.LINE_AA)

    bx, by = tx-4, ty+10
    bw, bh = 162, 72

    ov = img.copy()
    cv2.rectangle(ov,(bx,by),(bx+bw,by+bh),(0,0,0),-1)
    cv2.addWeighted(ov,0.58,img,0.42,0,img)
    cv2.rectangle(img,(bx,by),(bx+bw,by+bh),OR,2,cv2.LINE_AA)

    # inner thin border
    cv2.rectangle(img,(bx+3,by+3),(bx+bw-3,by+bh-3),OR2,1,cv2.LINE_AA)

    cv2.putText(img,"palm data::", (bx+7,by+23),
                cv2.FONT_HERSHEY_SIMPLEX,0.46,OR,1,cv2.LINE_AA)
    cv2.putText(img,f":{pdat:05.1f}%", (bx+7,by+58),
                cv2.FONT_HERSHEY_DUPLEX,0.82,OR,2,cv2.LINE_AA)

    # glitch offset on palm data
    if glitch:
        cv2.putText(img,f":{pdat:05.1f}%",
                    (bx+7+np.random.randint(-4,5), by+58),
                    cv2.FONT_HERSHEY_DUPLEX,0.82,
                    (30,30,200),1,cv2.LINE_AA)

    draw_cube(img, bx-60, by+36, t)
    draw_grid(img, bx-80, by+76)

    # small "TRACKING" label top-right
    cv2.putText(img,"[ TRACKING ACTIVE ]",(frame_w-210,25),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,OR,1,cv2.LINE_AA)

    # FPS
    fps_val = getattr(draw_hud,'fps',60)
    cv2.putText(img,f"FPS {fps_val:.0f}",(frame_w-90,45),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,DIM,1,cv2.LINE_AA)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    global lag_ring_angle, lag_ring_vel, lag_inner_angle, lag_inner_vel
    global prev_palm_angle, palm_angle_smooth, smooth_pts, smooth_wrist

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    print("=========================================")
    print("  ULTRA ROBOT HAND TRACKER  |  Q = quit")
    print("=========================================\n")

    t, prev, ts_ms = 0.0, time.time(), 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        now    = time.time(); dt = now-prev; prev = now; t += dt
        ts_ms += max(int(dt*1000),1)
        dt     = min(dt, 0.1)   # clamp

        # cyberpunk darkening + slight blue tint
        dark = np.zeros_like(frame)
        dark[:,:,0] = 5   # faint blue channel
        frame = cv2.addWeighted(frame, 0.70, dark, 0.30, 0)

        # subtle scan line
        draw_scanline(frame, t, h, w)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        raw_pts = None
        raw_lms = None
        if result.hand_landmarks:
            raw_lms = result.hand_landmarks[0]
            raw_pts = [(int(lm.x*w), int(lm.y*h)) for lm in raw_lms]

        if raw_pts and len(raw_pts)==21:
            # ── Smooth landmarks (lag effect) ──
            if smooth_pts is None:
                smooth_pts = [list(p) for p in raw_pts]
            else:
                for i in range(21):
                    smooth_pts[i][0] = int(smooth_pts[i][0]*(1-SMOOTH_ALPHA) + raw_pts[i][0]*SMOOTH_ALPHA)
                    smooth_pts[i][1] = int(smooth_pts[i][1]*(1-SMOOTH_ALPHA) + raw_pts[i][1]*SMOOTH_ALPHA)
            pts = [tuple(p) for p in smooth_pts]

            # ── Smooth wrist for arm ──
            if smooth_wrist is None:
                smooth_wrist = list(raw_pts[0])
            else:
                smooth_wrist[0] = int(smooth_wrist[0]*(1-ARM_ALPHA)+raw_pts[0][0]*ARM_ALPHA)
                smooth_wrist[1] = int(smooth_wrist[1]*(1-ARM_ALPHA)+raw_pts[0][1]*ARM_ALPHA)

            # ── Palm rotation angle for lag rings ──
            cur_angle = math.degrees(math.atan2(
                -(pts[9][1]-pts[0][1]), pts[9][0]-pts[0][0])) % 360

            if prev_palm_angle is None:
                prev_palm_angle = cur_angle

            # angular delta (wrap-aware)
            delta = cur_angle - prev_palm_angle
            if delta > 180: delta -= 360
            if delta < -180: delta += 360
            prev_palm_angle = cur_angle

            # spring-damper for lag rings
            # outer ring lags more
            target_vel = delta / max(dt, 0.001) * 0.6
            lag_ring_vel  += (target_vel - lag_ring_vel) * dt * 3.0
            lag_ring_vel  *= 0.88   # damping
            lag_ring_angle += lag_ring_vel * dt + t*25  # also auto-spin

            # inner ring lags less
            lag_inner_vel += (target_vel*1.4 - lag_inner_vel) * dt * 6.0
            lag_inner_vel *= 0.82
            lag_inner_angle += lag_inner_vel * dt + t*55

            draw_skeleton(frame, pts, raw_lms, t,
                          lag_ring_angle, lag_inner_angle, w, h)
            rot  = get_rotation(pts)
            pdat = get_palm_data(pts)
            draw_hud.fps = 1/max(dt,1e-9)
            draw_hud(frame, rot, pdat, pts[0][0], pts[0][1], t, w, h)

        else:
            # clear trails when hand lost
            for k in tip_trails: tip_trails[k].clear()
            smooth_pts   = None
            smooth_wrist = None
            cv2.putText(frame,"[ SCANNING FOR HAND... ]",(30,55),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,DIM,1,cv2.LINE_AA)
            cv2.putText(frame,"[ TRACKING ACTIVE ]",(w-210,25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,OR,1,cv2.LINE_AA)

        cv2.imshow("ULTRA ROBOT HAND TRACKER", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
