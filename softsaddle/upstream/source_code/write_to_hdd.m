function write_to_hdd(matrix, working_path, f_name)

% [~, col] = size(matrix);
% if col>2 % raise an error and stop
%     err = MException('ResultChk:OutOfRange', ...
%         'Array has more than 2 colums. Reformat');
%     throw(err)
% else % prepare the file to write
% %     fileID = fopen([working_path, f_name],'wt');  % 'wt' for writing in text mode
% end
% 
% if col==1
%         fprintf(fileID,'%f\n',matrix);  % writes matrix to a txt file
% elseif col==2
%         fprintf(fileID,'%f %f\r\n',matrix);
% end
%         % close the file
%         fclose(fileID); % close the file that keeps matrix

     my_file = fullfile(working_path, f_name);
     %disp(my_file)
     
% Begin For error debugging
% save(my_path, 'my_file', '-ASCII')     
% End For error debugging
     
     dlmwrite(my_file, matrix, 'delimiter', '\t', 'precision', 9)
       
