function [img_corr, RRMSE, target, C, D, P] = inpainting(img_clean, img_corr_orig, target)

img_corr = img_corr_orig;
[size_x , size_y] = size(img_corr);

block_size = 7;
width = (block_size-1)/2;
search_radius = 20;
alpha = 1;

iter = 0;
iter_max = 10000;


num_corr = 0;
C = 1-target;
D = zeros(size(C));

RRMSE = zeros(1, iter_max);

%hsize = 11;
%sigma = 8;
%gaus_filt = fspecial('gaussian',hsize,sigma);


while(iter < iter_max && sum(sum(target)) > 0)
    iter = iter + 1;
    D = zeros(size(C));

    %[Gmag,~] = imgradient(target);
    [Gmag,~] = imgradient(target, 'intermediate');
    
    %[T_X, T_Y] = gradient(target);
    [T_X, T_Y] = imgradientxy(target, 'intermediate');

    [I_X, I_Y] = imgradientxy(img_corr);

    boundary = Gmag.*target;

    C_new = C;
    
    [b_i, b_j] = find(boundary);
    
    for index = 1: size(b_i)  
        i = b_i(index);
        j = b_j(index);
        img_grad = [I_X(i,j), I_Y(i,j)];
        targ_grad = [T_X(i,j), T_Y(i,j)];

        D(i,j) = abs(dot(img_grad, targ_grad))/alpha;

        C_P = zeros(size(img_corr));
        C_P(i-width: i+width, j-width: j+width) = 1;
        C_new(i,j) = sum(sum(C .* C_P .* (1-target)))/(block_size^2);
        %disp(num2str(i) + " " + num2str(j));
%               temp = imfilter((C .* C_P .* (1-target)/(block_size^2)), gaus_filt);
%               C_new(i,j) = temp(i,j);
    end
    
    C = C_new;
    P = C.* D;
    
    A = boundary.*P;
    [y,~] = max(A);
    [~,column] = max(y);
    [~,row] = max(A(:,column));
    
    if (target(row, column) == 1)
        min_norm = inf;
        for i = max(1+width, row-search_radius) : min(row+search_radius, size_x-width)
            for j = max(1+width, column-search_radius) : min(column+search_radius, size_y-width)
                if (sum(sum(target(i-width: i+width, j-width: j+width))) == 0)
                    filt = 1-target;
                    img_patch = img_corr(i-width: i+width, j-width: j+width).* filt(row-width: row+width, column-width: column+width);
                    corr_patch = img_corr(row-width: row+width, column-width: column+width) .* filt(row-width: row+width, column-width: column+width);
                    norm = sum(sum(((img_patch-corr_patch).^2)));
                    if (norm < min_norm)
                        min_norm = norm;
                        patch_i = i;
                        patch_j = j;
                    end
                end
            end
        end
        
        img_corr(row, column) = img_corr(patch_i, patch_j);
        %disp("Pixel at (" + num2str(row) + "," + num2str(column) + ") replaced by (" + num2str(patch_i) + "," + num2str(patch_j) + ")");
        target(row, column) = 0;
        C(row, column) = 1;
        RRMSE(iter) = sqrt(sum(sum((img_clean-img_corr).^2)));

        num_corr = num_corr + 1;
    else
        disp("INCORRECT Behaviour. Minimum found at Row: " + num2str(row) + " and Column: " + num2str(column));
    end
    
%    
%     plot(1:iter, RRMSE(1:iter));
%     ylabel("RRMSE");
%     xlabel("Iteration");
%     drawnow;
    imshow(img_corr, []);
    drawnow;
   
end

end