%###names of input and output folders###
%input folder in the format:
%Dataset |-> NCSU-CUB_Foram_Images_G-bulloides
%        |->   .
%        |->   .
%        |->   .
%        |-> NCSU-CUB_Foram_Images_Others    

path = {'NCSU-CUB_Foram_Images_G-bulloides','NCSU-CUB_Foram_Images_G-ruber','NCSU-CUB_Foram_Images_G-sacculifer','NCSU-CUB_Foram_Images_N-incompta','NCSU-CUB_Foram_Images_N-pachyderma','NCSU-CUB_Foram_Images_Others'};
outF = 'clusterIMG';

%create output folder
mkdir(outF);
%start of main loop, goes through all folders of the dataset 

for K = 1 : length(path)

    %create datastore of the selected folder
    imB = imageDatastore(strcat('Dataset/',path{K}), ...
                         'IncludeSubfolders', true, ...
                         'LabelSource','foldernames');
    
    %create new folder in the selected output folde that has the same name of the input folder                  
    mkdir(outF,path{K}); 
    
    %go through each image in the dataStore
    I = 1;
    while I < length(imB.Labels)                

        %imgR and imgC stands for imgRows and imgColumns
        [imgR, imgC] = size(readimage(imB,I));
        px = zeros(imgR,imgC,16);
        %this loop goes through 16 images at a time and stores the value of
        %each pixel in a 3D matrix 
        for J = 1 : 16
            img = readimage(imB,I);
            for R = 1 : imgR
               for C = 1 : imgC
                   px(R,C,J) = img(R,C);
               end
            end
            I = I + 1;
        end

        img90 = zeros(imgR,imgC);
        img50 = zeros(imgR,imgC);
        img10 = zeros(imgR,imgC);
        
        %this loop processes 10th, 50th and 90th percentile of each group
        %of 16 pixel gathered in the previous step
        parfor R = 1 : imgR
            for C = 1 : imgC
                %Gets all the values in the third dimension given the row R
                %and column C => it is a 1-D vector
                pix = px(R,C,:);
                pix = squeeze(pix);
                [output,c] = kmeans(pix, 3);
                % first column = avg, second column = count
                %{
                avg = zeros(3,2);
                for k = 1 : size(pix)
                    avg(output(k),2) = avg(output(k),2) +1;
                    avg(output(k),1) = avg(output(k),1) + pix(k);
                end
                for k = 1 : 3
                    avg(k,1) = avg(k,1) / avg(k,2);
                end
                %sorting rows based on the number of votes for each class
                avg = sortrows(avg, 2);
                img90(R,C) = avg(3,1);
                img50(R,C) = avg(2,1);
                img10(R,C) = avg(1,1);
                %}
                centr = zeros(3,2);
                for i = 1:3
                    centr(i,1) = c(i);
                    centr(i,2) = sum(output==i);
                end 
                centr = sortrows(centr,2);                    
                %If the lightest color is lower than a threshold, then the
                %pixel will be black. We remove unnecessary noise                
                if centr(1,1) < 25
                    centr2 = zeros(3)
                else
                    for i = 1:3
                        centr2(i) = centr(i,1);
                    end                    
                end               
                img90(R,C) = centr2(3);
                img50(R,C) = centr2(2);
                img10(R,C) = centr2(1);
                
            end
        end
        %since we still have greyscale images we need to use uint8 format
        %for pixels values
        img90 = uint8(img90);
        img50 = uint8(img50);
        img10 = uint8(img10);
        
        %every matrix obtained this way is used as a channel in the RGB image
        %and is than saved in the new folder
        imgO = cat(3,img10,img50,img90);
        nome = strcat(outF,'/',path{K},'/',char(imB.Labels(I-1)),'.png');        
        imwrite(imgO,nome);
        keyboard
    end
end

