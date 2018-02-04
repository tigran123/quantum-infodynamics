ffmpeg -y -i tachyon.mp4 -i tardyon.mp4 -i luxon.mp4 -filter_complex '[0:v] pad=2*iw:2*ih [tachyon]; [tachyon][1:v] overlay=main_w/2:0,overlay=0:main_h/2' out.mp4
