% cut the video

v = VideoReader('test.mp4');

out_mp4_name = 'test_ac1.mp4';
v_o = VideoWriter(out_mp4_name, 'MPEG-4');

open(v_o);

now_num = 0;

while hasFrame(v)
    video = readFrame(v);
    video_part = video(450:768, 450:850, :);
    now_num = now_num + 1;
    if mod(now_num, 2)==0
        writeVideo(v_o, video_part);
    end
end

close(v_o);