import av, sys, json, os
root='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image'
print("=== TEST/VAL VIDEOS (cat1_test_set_public) ===")
for v in range(1,8):
    p=f'{root}/cat1_test_set_public/{v}_fps1.mp4'
    c=av.open(p); s=c.streams.video[0]
    n=s.frames
    print(f'  {v}: {s.codec_context.width}x{s.codec_context.height} frames={n} fps={float(s.average_rate):.3f} dur={float(s.duration*s.time_base) if s.duration else -1:.1f}s')
    c.close()
print("=== TRAIN VIDEOS ===")
import glob
vids=sorted(glob.glob(f'{root}/surgvu24_videos_only/surgvu24/*/*.mp4'))
print('n videos', len(vids))
for p in vids[:5]:
    c=av.open(p); s=c.streams.video[0]
    print(f'  {os.path.basename(p)}: {s.codec_context.width}x{s.codec_context.height} frames={s.frames} fps={float(s.average_rate):.2f} dur={float(s.duration*s.time_base) if s.duration else -1:.1f}s')
    c.close()
