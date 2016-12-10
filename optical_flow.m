function [U, V, elasped_time] = optical_flow(file1, file2, num_it, avg_window, alpha)
%% Setup
if ~exist('num_it', 'var')
    num_it = 100;
end
if ~exist('avg_window', 'var')
    avg_window = 5;
end
if ~exist('alpha', 'var')
    alpha = 1;
end

if ~exist('file1', 'var') || ~exist('file2', 'var')
    [filename1, filepath1] = uigetfile({'*.*'});
    [filename2, filepath2] = uigetfile({'*.*'});
    file1 = strcat(filepath1, filename1);
    file2 = strcat(filepath2, filename2);
end

img1 =imread(file1);
img2 = imread(file2);
% Convert RGB to grayscale
if (numel(size(img1)) == 3)
    img1 = rgb2gray(img1);
end
if (numel(size(img2)) == 3)
    img2 = rgb2gray(img2);
end
A1 = im2double(img1);
A2 = im2double(img2);

% Calculate derivatives
I_t = A2 - A1;
[I_x, I_y] = imgradientxy(A1);

U = zeros(size(I_t));
V = zeros(size(I_t));
kernel = ones(avg_window) / avg_window^2;

%% Calculate
tic
for k = 1:num_it
    % calculate average with moving convolution
    U_avg = conv2(U, kernel, 'same');
    V_avg = conv2(V, kernel, 'same');
    % calculate (u, v) with optimized matrix operations
    C = (I_x .* U_avg + I_y .* V_avg + I_t) ./ (alpha^2 + I_x.^2 + I_y.^2);
    U = U_avg - I_x .* C;
    V = V_avg - I_y .* C;
end
elasped_time = toc;
%% Plot
quiver(flipud(U), flipud(V));
%img = computeColor(U, V);
%imshow(img);
%writeFlowFile(cat(3, U, V), 'flow10.flo');
end