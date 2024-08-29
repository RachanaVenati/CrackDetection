function CrackDetection
clc;
clear all;
close all;
image_input=imread("/Users/rachanavenatiicloud.com/Crack_detection/CrackDetection/training_images/crack_3.jpeg");%reading image input
preproc_image=preprocessing(image_input);%preprocessing for contrast streching.
binaryimage=imageThresholding(preproc_image);%task 2a)adaptive thresholding (binary image).
morph_image=imageMorphology(binaryimage);
labeled_img=ConnectedComponentAnalysis(morph_image);
image_paths = {
    '/Users/rachanavenatiicloud.com/Crack_detection/CrackDetection/training_images/crack_1.jpg','/Users/rachanavenatiicloud.com/Crack_detection/CrackDetection/training_images/crack_3.jpeg'
};
ground_truth_paths = {
    '/Users/rachanavenatiicloud.com/Downloads/ram1/SegmentationClass/crack_1.png', '/Users/rachanavenatiicloud.com/Downloads/ram/SegmentationObject/crack_3.png'
};

crack_detection_pipeline(image_paths,ground_truth_paths);
end

function prepro_img=preprocessing(image_input)
grayscaled_img=rgb2gray(image_input);%converting input image to grayscaled image
min_intensity=double(min(min(grayscaled_img)));
max_intensity=double(max(max(grayscaled_img)));
streched_image=(grayscaled_img-min_intensity)*(255/(max_intensity-min_intensity));
prepro_img=streched_image;
figure;
imshow(streched_image);
title("Contrast streched image");

end

function Binary_image = imageThresholding(image)%task 2a.)thresholding
    blockSize = 15; 
    C = 20;
    thresholded_img = adaptthresh(image, 0.5, 'NeighborhoodSize', blockSize, 'Statistic', 'mean', 'ForegroundPolarity', 'dark');
    thresholded_img = thresholded_img - (C / 255);%reducing the strictness of thresholding using c.
    Binary_image = imbinarize(image, thresholded_img);
    Binary_image = ~Binary_image;
    figure;
    imshow(Binary_image);
    title("Thresholding Result");
end

function morphlogical_img= imageMorphology(image)
dilate=strel('square',1);
erode=strel('disk',0);
close=strel('cube',1);
open=strel('square',2);
closing=imclose(image,close);
opening=imopen(closing,open);
errosion=imerode(opening,erode);
dilation=imdilate(errosion,dilate);
filling=imfill(dilation,'holes');
morphlogical_img=filling;
figure;
imshow(morphlogical_img);
title("Image after Morphology");
end

function labeled_img = ConnectedComponentAnalysis(binary_image)
    [rows, cols] = size(binary_image);
    labeled_img = zeros(rows, cols);
    
    current_label = 0;
    connectivity = [0 1; 1 0; 1 1; 1 -1; -1 1; -1 -1; -1 0; 0 -1];
    
    for row = 1:rows
        for col = 1:cols
            if binary_image(row, col) == 1
                neighbors = [];
                for k = 1:size(connectivity, 1)
                    r = row + connectivity(k, 1);
                    c = col + connectivity(k, 2);
                    if r >= 1 && r <= rows && c >= 1 && c <= cols && labeled_img(r, c) > 0
                        neighbors = [neighbors, labeled_img(r, c)];
                    end
                end
                
                if isempty(neighbors)
                    current_label = current_label + 1;
                    labeled_img(row, col) = current_label;
                else
                    min_label = min(neighbors);
                    labeled_img(row, col) = min_label;
                    
                    for neighbor_label = neighbors
                        if neighbor_label > min_label
                            labeled_img(labeled_img == neighbor_label) = min_label;
                        end
                    end
                end
            end
        end
    end
    
    labeled_img = relabelComponents(labeled_img);
    
    figure;
    imshow(label2rgb(labeled_img));
    title('Connected Components Labeled Image');
end

function mergeLabels(labeled_img, old_label, new_label)
    labeled_img(labeled_img == old_label) = new_label;
end

function [relabel_img, num_labels] = relabelComponents(label_img)
    unique_labels = unique(label_img(label_img > 0));
    num_labels = length(unique_labels);
    relabel_img = zeros(size(label_img));
    
    for i = 1:num_labels
        relabel_img(label_img == unique_labels(i)) = i;
    end
end

function crack_detection_pipeline(image_paths, ground_truth_paths)
    num_images = length(image_paths);
    num_train_images = 1; % Adjust this number as needed
    num_test_images = num_images - num_train_images;
    
    % Initialize feature vectors and labels for training
    feature_vectors = [];
    labels = [];
    
    % Train on the first 'num_train_images' images
    for i = 1:num_train_images
        image_input = imread(image_paths{i});
        ground_truth = imread(ground_truth_paths{i});
        
        preproc_image = preprocessing(image_input);
        binaryimage = imageThresholding(preproc_image);
        morph_image = imageMorphology(binaryimage);
        labeled_img = ConnectedComponentAnalysis(morph_image);
        
        % Extract features and labels from training images
        [fvs, lbls] = extract_features(labeled_img, ground_truth, preproc_image);
        feature_vectors = [feature_vectors; fvs];
        labels = [labels; lbls];
    end
    
    % Train the SVM model using the extracted features
    svm_model = svm_classifier(feature_vectors, labels);
    
    % Test on the remaining images
    for i = num_train_images+1:num_images
        image_input = imread(image_paths{i}); % Corrected: Use current image path
        ground_truth = imread(ground_truth_paths{i});
        
        preproc_image = preprocessing(image_input);
        binaryimage = imageThresholding(preproc_image);
        morph_image = imageMorphology(binaryimage);
        labeled_img = ConnectedComponentAnalysis(morph_image);
        
        % Extract features from the test image
        ground_truth = imresize(ground_truth, size(labeled_img));
        [fvs, lbls] = extract_features(labeled_img, ground_truth, preproc_image);
       
        % Make predictions
        predictedLabels = predict(svm_model, fvs);
        
        % Calculate accuracy or IoU on test image
        accuracy_svm = sum(predictedLabels == lbls) / length(lbls);
        fprintf('Test Accuracy on Image %d: %.2f%%\n', i, accuracy_svm * 100);
        
        iouCrack = calculate_iou(ground_truth, labeled_img);
        disp(['IoU for Crack on Image ', num2str(i), ': ', num2str(iouCrack)]);
    end
end
function [feature_vectors, labels] = extract_features(label_matrix, ground_truth, preproc_image)
    stats = regionprops(label_matrix, preproc_image, 'Area', 'Perimeter', 'Eccentricity', ...
        'MajorAxisLength', 'MinorAxisLength', 'ConvexArea');
    
    feature_vectors = [];
    labels = [];
    
    for i = 1:max(label_matrix(:))
        % Extract features
        area = stats(i).Area;
        perimeter = stats(i).Perimeter;
        eccentricity = stats(i).Eccentricity;
        circularity = (4 * pi * area) / (perimeter^2);
        aspect_ratio = stats(i).MajorAxisLength / stats(i).MinorAxisLength;
        solidity = area / stats(i).ConvexArea;
        
        % Create a feature vector for this region
        feature_vector = [area, perimeter, eccentricity, circularity, aspect_ratio, solidity];
        feature_vectors = [feature_vectors; feature_vector];
        
        % Determine label based on ground truth overlap
        region_mask = (label_matrix == i);
        disp(['Size of region_mask: ', num2str(size(region_mask))]);
        region_area_ground_truth = sum(ground_truth(region_mask));
        disp(['Size of ground_truth: ', num2str(size(ground_truth))]);
        if region_area_ground_truth > 0
            label = 1; % For crack
        else
            label = 0; % No crack
        end
        
        labels = [labels; label];
    end
    
    % Check if both classes (1 and 0) are present in the labels
    if length(unique(labels)) < 2
        error('Both classes (crack and no-crack) must be present in the labels.');
    end
end
function accuracy_svm = svm_classifier(feature_vectors, labels)
    % Ensure the labels array is not empty and contains both classes
    if isempty(labels)
        error('The labels array is empty.');
    end
    
    unique_labels = unique(labels);
    if length(unique_labels) < 2
        error('The labels array does not contain both classes. Found classes: %s', mat2str(unique_labels));
    end
    
    % Display the initial distribution of labels
    disp('Initial Label distribution:');
    disp(tabulate(labels));
    
    % Check for zero variance in the features
    feature_variances = var(feature_vectors);
    disp('Feature Variances:');
    disp(feature_variances);
    
    % Remove features with zero variance
    zero_variance_mask = feature_variances == 0;
    if all(zero_variance_mask)
        error('All features have zero variance.');
    end
    
    feature_vectors = feature_vectors(:, ~zero_variance_mask);
    
    % Split the data into training and testing sets
    split_ratio = 0.8;
    x = randperm(length(labels));
    num_train_samples = round(split_ratio * length(labels));
    
    % Partition the data into training and testing sets
    train_data = feature_vectors(x(1:num_train_samples), :);
    train_labels = labels(x(1:num_train_samples));
    
    test_data = feature_vectors(x(num_train_samples + 1:end), :);
    test_labels = labels(x(num_train_samples + 1:end));
    
    % Train the SVM model directly on the unscaled data
    svm_model = fitcsvm(train_data, train_labels, 'KernelFunction', 'linear');
    
    % Make predictions on the test data
    predictions_svm = predict(svm_model, test_data);
    
    % Calculate the accuracy of the model on the test data
    correct_predictions_svm = sum(predictions_svm == test_labels);
    total_samples_svm = length(test_labels);
    accuracy_svm = correct_predictions_svm / total_samples_svm;
    
    % Display the test accuracy
    fprintf('Test Accuracy using SVM without scaling: %.2f%%\n', accuracy_svm * 100);
end
function iouCrack = calculate_iou(ground_truth, finalBinaryImage)

    ground_truth_bw = imbinarize(ground_truth);

    ground_truth_resized = imresize(ground_truth_bw, size(finalBinaryImage)); 
    
    intersectionCrack = sum(ground_truth_resized(:) & finalBinaryImage(:));
    unionCrack = sum(ground_truth_resized(:) | finalBinaryImage(:)); 
    
    iouCrack = intersectionCrack / unionCrack;
end