function [config,sanned_param] = readINI(filename)
    % Reads an INI file into a MATLAB structure
    % Input:
    %   filename - name of the INI file to read
    % Output:
    %   config - structure containing parsed INI file data

    % Open the file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end

    config = struct(); % Initialize the configuration structure
    currentSection = ''; % Track the current section
    sanned_param = [];
    while ~feof(fid)
        line = strtrim(fgets(fid)); % Read a line and trim whitespace

        % Skip empty lines and comments
        if isempty(line) || startsWith(line, ';') || startsWith(line, '#')
            continue;
        end

        % Check if the line is a section header
        if startsWith(line, '[') && endsWith(line, ']')
            currentSection = line(2:end-1); % Extract section name
            config.(currentSection) = struct(); % Initialize section in structure
            continue;
        end

        % Parse key-value pairs
        if contains(line, '=')
            keyValue = split(line, '=', 2); % Split key and value
            key = strtrim(keyValue{1});
            value = strtrim(keyValue{2});

            % Remove inline comments
            commentIdx = find(value == ';', 1);
            if ~isempty(commentIdx)
                value = strtrim(value(1:commentIdx-1));
            end

            % Try to interpret numeric or logical values
            if ismember(lower(value), {'true', 'false'})
                value = strcmpi(value, 'true'); % Convert to logical
            elseif contains(value, ',')
                % Handle comma-separated lists
                value = str2double(split(value, ','));
                sanned_param = {currentSection,key};
            else
                numericValue = str2double(value);
                if ~isnan(numericValue)
                    value = numericValue; % Convert to numeric if possible
                end
            end

            % Add key-value pair to the current section
            config.(currentSection).(key) = value;
        end
    end

    % Close the file
    fclose(fid);
end
