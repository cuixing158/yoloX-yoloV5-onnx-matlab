%% yoloX-tiny.onnx自定义导出的onnx成功运行

onnxfile = "./onnxPretrainedModels/yolox_tiny.onnx";
imds = imageDatastore('./images/dog.jpg');
customYoloXFcnName = 'yoloxFcn';

%% Parameters matching the model file onnx
inferenceSize = [416,416];% Must match the size of the onnx input
strides = [8,16,32];% Downsampling rate
throushHold = 0.1;
nmsThroushHold = 0.4;
classesNames = categorical(readlines('coco.names','EmptyLineRule','skip'));
colors = randi(255,length(classesNames),3);
params = importONNXFunction(onnxfile,customYoloXFcnName);


%% detect
for i = 1:length(imds.Files)
    oriImg = imread(imds.Files{i});
    [oriHight,oriWidth,~] = size(oriImg);
    img = imresize(oriImg,inferenceSize);
    img = rescale(img,0,1);% 转换到[0,1]
    img = permute(img,[3,1,2]);% matlab defaults to column order over row order
    img = dlarray(reshape(img,[1,size(img)])); % n*c*h*w，[0,1],RGB顺序
    if canUseGPU()
        img = gpuArray(img);
    end
    out = feval(customYoloXFcnName,img,params,...
        'Training',false,...
        'InputDataPermutation','none',...
        'OutputDataPermutation','none');% or call the function directly
    outFeatures = yoloxDecode(out,oriHight,oriWidth,inferenceSize,strides);
    
    %% 阈值过滤+NMS处理

    scores = outFeatures(:,5);
    outFeatures = outFeatures(scores>throushHold,:);
    
    allBBoxes = outFeatures(:,1:4);
    [maxScores,indxs] = max(outFeatures(:,6:end),[],2);
    allScores = maxScores;
    allLabels = classesNames(indxs);
    
    % NMS非极大值抑制
    drawImg = oriImg;
    if ~isempty(allBBoxes)
        [bboxes,scores,labels] = selectStrongestBboxMulticlass(allBBoxes,...
            allScores,allLabels,...
            'RatioType','Min','OverlapThreshold',nmsThroushHold);
        annotations = string(labels) + ": " + string(scores);
        [~,ids] = ismember(labels,classesNames);
        color = colors(ids,:);
        drawImg = insertObjectAnnotation(drawImg,...
            'rectangle',bboxes,cellstr(annotations),...
            'Color',color,...
            'LineWidth',3);
    end
    imshow(drawImg);
    drawnow
end



%% support function

function outPutFeatures = yoloxDecode(featuremaps,oriHight,oriWidth,...
    inferenceSize,strides)
% 功能：yolox解码
% 输入：
%     featuremaps: 官方格式的onnx，bs*numNeutrals*(4+1+numClasses)，未解码状态
%
% 注意：
% 1、检测目标类别数量为coco中的80
%  其他onnx模型格式类推，大同小异
%
% reference:
% https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime
%
% author: cuixingxing
% email:cuixingxing150@gmail.com
% 2021.7.30创建
%
arguments
    featuremaps (1,:,85) 
    oriHight (1,1) double
    oriWidth (1,1) double
    inferenceSize (1,2) double = [416,416]% onnx输入网络图像大小，正方形图像输入
    strides (1,:) double = [8,16,32]% onnx模型下采样率
end

%% assert
scaledX = inferenceSize(1)./oriWidth;
scaledY = inferenceSize(2)./oriHight;
[bs,numNeu,numF] = size(featuremaps);
featuresMapsSizes  = inferenceSize(1)./strides;
assert(numNeu==sum(featuresMapsSizes.^2));

%% decode
numberFeaturemaps = length(strides);
outPutFeatures = [];
endIdx = 0;
for i = 1:numberFeaturemaps
    startIdx = endIdx+1;
    endIdx = endIdx+featuresMapsSizes(i)^2;% 当前特征图神经元个数
    output = featuremaps(:,startIdx:endIdx,:);% bs*[h*w]*[5+nc]大小
    [X,Y] = meshgrid(0:featuresMapsSizes(i)-1);
    grid = cat(3,X,Y);% h*w*2
    grid = permute(grid,[2,1,3]);
    grid = reshape(grid,1,[],2);% 1*numF*2
    currentFeatureMap = output;
    currentFeatureMap(:,:,1:2) = (output(:,:,1:2)+grid).*strides(i);
    currentFeatureMap(:,:,3:4) = exp(output(:,:,3:4)).*strides(i);
    
    if isempty(outPutFeatures)
        outPutFeatures = currentFeatureMap;
    else
        outPutFeatures = cat(2,outPutFeatures,currentFeatureMap);% bs*M*(5+nc)
    end
end
%% 坐标转换到原始图像上

outPutFeatures = extractdata(outPutFeatures);% bs*M*(5+nc) ,为[x_center,y_center,w,h,Pobj,p1,p2,...,pn]
outPutFeatures(:,:,[1,3]) = outPutFeatures(:,:,[1,3])./scaledX;% x_center,width
outPutFeatures(:,:,[2,4]) = outPutFeatures(:,:,[2,4])./scaledY;% y_center,height
outPutFeatures(:,:,1) = outPutFeatures(:,:,1) -outPutFeatures(:,:,3)/2;%  x
outPutFeatures(:,:,2) = outPutFeatures(:,:,2) -outPutFeatures(:,:,4)/2; % y

outPutFeatures = squeeze(outPutFeatures); % 如果是单张图像检测，则输出大小为M*(5+nc)，否则是bs*M*(5+nc)
if(canUseGPU())
    outPutFeatures = gather(outPutFeatures); % 推送到CPU上
end
end