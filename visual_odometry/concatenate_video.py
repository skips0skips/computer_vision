from moviepy.editor import VideoFileClip, concatenate_videoclips


clip_1 = VideoFileClip("train_data\0.avi")
clip_2 = VideoFileClip("train_data\1.avi")
clip_3 = VideoFileClip("train_data\2.avi")
final_clip = concatenate_videoclips([clip_1,clip_2,clip_3])
final_clip.write_videofile("final.avi")