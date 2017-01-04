image_type= 'IR_Reg';
phase = 'test';
image_db = '/home/shuoliu/Research/IROD/code/fast-rcnn/data/sample_data';
file_path = [image_db '/Train_Test/' image_type '/' phase '.txt'];
disp(file_path);
image_filenames = textread(file_path, '%s', 'delimiter', '\n');
for i = 1:length(image_filenames)
    if exist([image_db '/Imagery/' image_type '/images/' image_filenames{i} '.png'], 'file') == 2
        image_filenames{i} = [image_db '/Imagery/' image_type '/images/' image_filenames{i} '.png'];
    end
end
selective_search_rcnn(image_filenames, [phase '.mat']);
