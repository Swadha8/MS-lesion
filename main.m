index = 100;
fin = fopen('Datasets/t1_icbm_normal_1mm_pn3_rf20.rawb','r');
x = 181;
y = 217;
z = 181;
I_1 = fread(fin, x*y*z, 'uint8=>uint8');
I_1 = reshape(I_1, [x y z]);
img_orig = I_1(:,:,index)';

fin = fopen('Datasets/t1_ai_msles2_1mm_pn3_rf20.rawb','r');
x = 181;
y = 217;
z = 181;
I = fread(fin, x*y*z, 'uint8=>uint8');
I = reshape(I, [x y z]);
img_corr = I(:,:,index)';

target = img_orig - img_corr;
threshold = 10;

target(target > threshold) = 255.0;
target(target <= threshold) = 0.0;
target = double(target/255.0);

img_orig = double(img_orig)/max(max(double(img_orig)));
img_corr = double(img_corr)/max(max(double(img_corr)));

[img_inpainted, RRMSE, ~, C, D, P] = inpainting(img_orig, img_corr, target);


figure;
subplot(1,4,1)
imshow(img_orig, []);
title("Original Image");
subplot(1,4,2)
imshow(img_corr, []);
title("Corrupted");
subplot(1,4,3)
imshow(target, []);
title("Target");
subplot(1,4,4)
imshow(img_inpainted, []);
title("Inpainted");
