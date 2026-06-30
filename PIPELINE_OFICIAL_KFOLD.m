% =========================================================================
% script k fold, entrene y testeo con dispositivos distintos
% =========================================================================

clear; clc;
rng(1); 
directorio = '../databases/'; 
archivos = dir(fullfile(directorio, '*RTT_RAW_W1S_*.mat'));
nombres_predictores = {'Ant1', 'Ant2', 'Ant3', 'Ant4'};
algoritmos = {'Fine Tree', 'Bagged Trees', 'SVM Gaussian', 'Optimizable Tree'};
resultados_finales = table();
K = 5; 

fprintf('Iniciando K-Fold (K=%d)...\n\n', K);

for i = 1:length(archivos)
    nombre_raw = archivos(i).name;
    ids = regexp(nombre_raw, '\d+', 'match');
    if isempty(ids) || length(ids{end}) < 2, continue; end
    id_pair = ids{end};
    id_tr = str2double(id_pair(1));
    id_te = str2double(id_pair(2));
    
    try
        temp = load(fullfile(directorio, nombre_raw));
        db = temp.database; 
        
        Data = array2table([ [db.trainingMacs; db.testMacs], [db.trainingLabels(:,1:2); db.testLabels(:,1:2)] ],'VariableNames', {'Ant1','Ant2','Ant3','Ant4','PosX','PosY'});
        Data.ID_Disp = [repmat(id_tr, height(db.trainingLabels), 1); repmat(id_te, height(db.testLabels), 1)];
        
        puntos_unicos = unique(Data(:, {'PosX', 'PosY'}), 'rows');
        puntos_unicos = sortrows(puntos_unicos, {'PosX', 'PosY'});
        puntos_unicos.ID_Punto = (1:height(puntos_unicos))';
        
        indices_azar = randperm(height(puntos_unicos));
        puntos_unicos.Fold = zeros(height(puntos_unicos), 1);
        for f = 1:5
            puntos_unicos.Fold(indices_azar((f-1)*4 + 1 : f*4)) = f;
        end
        Data = innerjoin(Data, puntos_unicos, 'Keys', {'PosX', 'PosY'});
        
        for k = 1:K
            T_Train = Data(Data.Fold ~= k & Data.ID_Disp == id_tr, :);
            T_Test  = Data(Data.Fold == k & Data.ID_Disp == id_te, :);
            
            if isempty(T_Train) || isempty(T_Test), continue; end
            
            min_Tr = min(T_Train{:, nombres_predictores}, [], 1, 'omitnan');
            max_Tr = max(T_Train{:, nombres_predictores}, [], 1, 'omitnan');
            rng_Tr = max_Tr - min_Tr; rng_Tr(rng_Tr == 0) = 1;
            
            for col = 1:4
                ant = nombres_predictores{col};
                T_Train{:, ant} = (T_Train{:, ant} - min_Tr(col)) / rng_Tr(col);
                T_Test{:, ant}  = (T_Test{:, ant} - min_Tr(col)) / rng_Tr(col);
                T_Train.(ant)(isnan(T_Train.(ant))) = 1.5;
                T_Test.(ant)(isnan(T_Test.(ant)))   = 1.5;
            end
            
            for a = 1:length(algoritmos)
                alg = algoritmos{a};
                t_train = tic;
                switch alg
                case 'Fine Tree'
                    mX = fitrtree(T_Train, 'PosX', 'PredictorNames', nombres_predictores, 'MinLeafSize', 4);
                    mY = fitrtree(T_Train, 'PosY', 'PredictorNames', nombres_predictores, 'MinLeafSize', 4);
                case 'Bagged Trees'
                    mX = fitrensemble(T_Train, 'PosX', 'Method', 'Bag', 'NumLearningCycles', 100, 'PredictorNames', nombres_predictores);
                    mY = fitrensemble(T_Train, 'PosY', 'Method', 'Bag', 'NumLearningCycles', 100, 'PredictorNames', nombres_predictores);
                case 'SVM Gaussian'
                    rsX = iqr(T_Train.PosX); if rsX == 0, rsX = 1; end
                    rsY = iqr(T_Train.PosY); if rsY == 0, rsY = 1; end
                    mX = fitrsvm(T_Train, 'PosX', 'KernelFunction', 'gaussian', 'KernelScale', 0.5, 'BoxConstraint', rsX/1.349, 'Standardize', true, 'PredictorNames', nombres_predictores);
                    mY = fitrsvm(T_Train, 'PosY', 'KernelFunction', 'gaussian', 'KernelScale', 0.5, 'BoxConstraint', rsY/1.349, 'Standardize', true, 'PredictorNames', nombres_predictores);
                case 'Optimizable Tree'
                    mX = fitrtree(T_Train, 'PosX', 'OptimizeHyperparameters', 'auto', 'PredictorNames', nombres_predictores, 'HyperparameterOptimizationOptions', struct('ShowPlots',0,'Verbose',0));
                    mY = fitrtree(T_Train, 'PosY', 'OptimizeHyperparameters', 'auto', 'PredictorNames', nombres_predictores, 'HyperparameterOptimizationOptions', struct('ShowPlots',0,'Verbose',0));
            end
                
                pX = predict(mX, T_Test);
                pY = predict(mY, T_Test);
                
                err_vec = sqrt((T_Test.PosX - pX).^2 + (T_Test.PosY - pY).^2);
                resultados_finales = [resultados_finales; table({nombre_raw}, {alg}, k, mean(err_vec,'omitnan'), prctile(err_vec,90), toc(t_train), ...
                    'VariableNames', {'Archivo', 'Algoritmo', 'Split', 'ErrMedio_m', 'P90_m', 'Tiempo_s'})];
            end
        end
        fprintf('%s: K-Fold completado.\n', nombre_raw);
    catch ME
        fprintf('Error en %s: %s\n', nombre_raw, ME.message);
    end
end

% --- RESULTADOS ---
fprintf('\n--- TABLA DETALLADA (K-FOLD) --- \n');
disp(resultados_finales);
fprintf('\n--- RESUMEN FINAL POR ARCHIVO ---\n');
disp(groupsummary(resultados_finales, {'Archivo', 'Algoritmo'}, 'mean', {'ErrMedio_m'}));
beep;