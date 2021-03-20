%% 简单导入yolov5s.onnx进行预测（推理）,无需解码即可进行
model = './onnxPretrainedModels/yolov5s_noProcessPost.onnx';
inputSize = [640,640];
throushHold = 0.3;
nmsThroushHold = 0.5;

%% import model
params = importONNXFunction(model,'yolov5sfcn');
classesNames = categorical(readlines("coco.names"));
colors = randi(255,length(classesNames),3);

%% 摄像头视频流识别
cap = webcam();
player = vision.DeployableVideoPlayer();
image = cap.snapshot();
step(player, image);

while player.isOpen()
    image = cap.snapshot();
    [H,W,~] = size(image);
    img = imresize(image,inputSize);
    img = rescale(img,0,1);% 转换到[0,1]
    img = permute(img,[3,1,2]);
    img = dlarray(reshape(img,[1,size(img)])); % n*c*h*w，[0,1],RGB顺序
    if canUseGPU()
        img = gpuArray(img);
    end
    t1 = tic;
    [classes,boxes] = yolov5sfcn(img,params,...
        'Training',false,...
        'InputDataPermutation','none',...
        'OutputDataPermutation','none');
    fprintf('yolov5s预测耗时：%.2f 秒\n',toc(t1));% yolov5s大概0.4秒
    
    %% 阈值过滤+NMS
    if canUseGPU() 
        classes = gather(extractdata(classes));
        boxes = gather(extractdata(boxes));
    end
    [maxvalue,idxs] = max(classes,[],2);
    validIdxs = maxvalue>throushHold;
    % nms
    indexes = idxs(validIdxs);
    predictBoxes = boxes(validIdxs,:);
    predictScores = maxvalue(validIdxs);
    predictNames = classesNames(indexes);
    predictBboxes = [predictBoxes(:,1)*W-predictBoxes(:,3)*W/2,...
        predictBoxes(:,2)*H- predictBoxes(:,4)*H/2,...
        predictBoxes(:,3)*W,...
        predictBoxes(:,4)*H];
    [bboxes,scores,labels] = selectStrongestBboxMulticlass(predictBboxes,...
        predictScores,...
        predictNames,...
        'RatioType','Min',...
        'OverlapThreshold',nmsThroushHold);
    annotations = string(labels) + ": " + string(scores);
    [~,ids] = ismember(labels,classesNames);
    predictColors = colors(ids,:);
    showImg = insertObjectAnnotation(image,'rectangle',bboxes,...
        cellstr(annotations),...
        'color',predictColors,...
        'LineWidth',3);
    step(player,showImg);
end
release(player);
