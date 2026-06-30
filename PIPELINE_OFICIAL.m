% =========================================================================
% script pipeline, entrene y testeo con dispositivos distintos
% =========================================================================

clear; clc;
rng(1);
directorio = '../databases/'; 
archivos = dir(fullfile(directorio, '*RTT_RAW_RAW_*.mat'));
nombres_predictores = {'Ant1', 'Ant2', 'Ant3', 'Ant4'};
algoritmos = {'Fine Tree', 'Bagged Trees', 'SVM Gaussian', 'Optimizable Tree'};
resultados_finales = table();

fprintf('Iniciando Pipeline ...\n\n');

for i = 1:length(archivos)
    nombre_raw = archivos(i).name;
    try
        temp = load(fullfile(directorio, nombre_raw));
        f = fieldnames(temp);
        db = temp.(f{1}); 
        
        T_Train = array2table([db.trainingMacs, db.trainingLabels(:,1:2)], ...
            'VariableNames', {'Ant1','Ant2','Ant3','Ant4','PosX','PosY'});
        T_Test = array2table([db.testMacs, db.testLabels(:,1:2)], ...
            'VariableNames', {'Ant1','Ant2','Ant3','Ant4','PosX','PosY'});
        
        min_Train = min(T_Train{:, nombres_predictores}, [], 2, 'omitnan');
        min_Test  = min(T_Test{:, nombres_predictores}, [], 2, 'omitnan');
        T_Train{:, nombres_predictores} = T_Train{:, nombres_predictores} - min_Train;
        T_Test{:, nombres_predictores}  = T_Test{:, nombres_predictores} - min_Test;
        
        min_vals = min(T_Train{:, nombres_predictores}, [], 1, 'omitnan');
        max_vals = max(T_Train{:, nombres_predictores}, [], 1, 'omitnan');
        range_vals = max_vals - min_vals; range_vals(range_vals == 0) = 1;
        
        for col = 1:4
            ant = nombres_predictores{col};
            T_Train{:, ant} = (T_Train{:, ant} - min_vals(col)) / range_vals(col);
            T_Test{:, ant}  = (T_Test{:, ant} - min_vals(col)) / range_vals(col);
            T_Train.(ant)(isnan(T_Train.(ant))) = 1.5;
            T_Test.(ant)(isnan(T_Test.(ant))) = 1.5;
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
            
            tiempo = toc(t_train);
            pX = predict(mX, T_Test); pY = predict(mY, T_Test);
            dist = sqrt((T_Test.PosX - pX).^2 + (T_Test.PosY - pY).^2);
            
            resultados_finales = [resultados_finales; table({nombre_raw}, {alg}, mean(dist,'omitnan'), prctile(dist,90), tiempo, ...
                'VariableNames', {'Archivo', 'Algoritmo', 'ErrMedio_m', 'P90_m', 'Tiempo_s'})];
        end
        fprintf(' %s completado.\n', nombre_raw);
    catch ME
        fprintf('Error en %s: %s\n', nombre_raw, ME.message);
    end
end
disp(groupsummary(resultados_finales, {'Archivo', 'Algoritmo'}, 'mean', {'ErrMedio_m', 'P90_m'}));
beep;