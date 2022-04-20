set /p in_dir="Enter input directory: "
set /p out_dir="Enter output directory: "
set /p choice="Enter 0 for running on image or 1 for running on video: "
set /p debug="Enter 1 for debug mode on image or 0 for none: "

python main.py %in_dir% %out_dir% %choice% %debug%