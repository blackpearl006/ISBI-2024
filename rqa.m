class_ = 'healthy';

% Read tau_dim table
tau_dim = readtable(sprintf('C:\\Users\\test_auk\\Documents\\Data\\fMRI\\merged tau and dim\\CB_%s.csv', class_));

% Get a list of all files in the specified directory
filePattern = sprintf('C:\\Users\\test_auk\\Documents\\Data\\fMRI\\ROI CSV files\\cerebellum_ts\\%s\\*.csv', class_);
files = dir(filePattern);

for fileIndex = 1:length(files)
    % Read the current CSV file
    filePath = fullfile(files(fileIndex).folder, files(fileIndex).name);
    subj = readtable(filePath);

    % Obtain subj_name from the directory
    [~, subjName, ~] = fileparts(filePath);

    % Loop through ROIs
    for roi = 1:18
        matchingRow = tau_dim(strcmp(tau_dim.subject, subjName) & tau_dim.ROI == roi, :);

        if ~isempty(matchingRow)
            roiData = subj(roi, :);
            y = table2array(roiData);
            xx(roi, :) = y;
            mm(roi) = matchingRow.DIM;
            tautau(roi) = matchingRow.Tau;

            if strcmp(threshold_calculation, 'fan') || strcmp(threshold_calculation, 'var')
                ee(roi) = 0.07;
            else
                ee(roi) = 1;
            end
        else
            disp('Matching row not found');
        end
    end

    % Initialize an empty cell array to store RQA measures
    a = cell(1, size(xx, 1));

    for k = 1:size(xx, 1)
        clear x m tau e w ws R r
        x = xx(k, :);
        m = mm(k);
        tau = tautau(k);
        timespanDiff = tau * (m - 1);
        xVec = embed(x, m, tau);
        e = ee(k);
        R = rp(xVec, e, threshold_calculation, norm);
        r = rqa(R, l_min, theiler);
        a{k} = r;
    end

    % Create a table with subject column as subjName and RQA measures
    resultTable = table(repmat({subjName}, size(xx, 1), 1), a', 'VariableNames', {'subject', 'RQA_measures'});

    % Save the table to a CSV file
    outputFilePath = 'C:\\Users\\test_auk\\Documents\\output_rqa.csv';
    writetable(resultTable, outputFilePath);
end
