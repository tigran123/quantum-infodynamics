A typical run may look like this


$ rm -rf frames ; mkdir frames ; python3 psi.py && ffmpeg -y -r 25 -f image2 -i frames/%04d.png -f mp4 -q:v 0 -vcodec libx264 -r 25 test.mp4 && mpv -fs -loop test.mp4
