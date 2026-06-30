% =========================================================================
% calcula el knn con configuracion 5 de las tablas de tablasfold.m 
% =========================================================================
addpath('metodos'); 
metodos = {'method_knn_plainrtt2025', 'method_wknnid_plainrtt2025', 'method_wknnsid_plainrtt2025'};
metricas = {'distancem_cityblock', 'distance_euclidean', 'distancem_minkowsky3', ...
            'distancem_cosine', 'distancem_sorensen', 'distancem_neyman'}; 
k_list = 1:2:51; 

archivos = fieldnames(dataset_completo);
resumen_final = []; 

fprintf('🚀 Iniciando experimento masivo... ¡Prepara el café, tiguer!\n');
tic_global = tic; 

for a = 1:length(archivos)
    for i = 1:5
        db_tr = dataset_completo.(archivos{a}).splits{i}.entreno;
        db_te = dataset_completo.(archivos{a}).splits{i}.test;
        db_tr.trainingMacs = db_tr.trainingMacs - min(db_tr.trainingMacs, [], 2, 'omitnan');
        db_te.testMacs = db_te.testMacs - min(db_te.testMacs, [], 2, 'omitnan');
        db_total = db_tr; db_total.testMacs = db_te.testMacs; db_total.testLabels = db_te.testLabels;
        
        tic_split = tic; 
        mejor_mae = inf; campeon = struct();
        
        for f = metodos, for m = metricas, for k = k_list
            try
                [~, res] = evalc(sprintf('%s(db_total, %d, ''rtt'', NaN, 100000, ''%s'')', f{1}, k, m{1}));
                if isfield(res, 'error')
                    mae = mean(res.error(:,1), 'omitnan');
                    if mae < mejor_mae
                        mejor_mae = mae;
                        campeon.mae = mae;
                        campeon.p90 = prctile(res.error(:,1), 90);
                        campeon.conf = sprintf('%s | K=%d | %s', f{1}, k, m{1});
                    end
                end
            catch, continue; end
        end, end, end
        
        tiempo_split = toc(tic_split);
        fprintf('%s | Split %d | MAE: %.2fm | P90: %.2fm | T: %.2fs\n', ...
            archivos{a}, i, campeon.mae, campeon.p90, tiempo_split);
        
        resumen_final = [resumen_final; {archivos{a}, i, campeon.mae, campeon.p90, tiempo_split, campeon.conf}];
    end
end

fprintf('\n\n======================================================\n');
fprintf('TABLA RESUMEN FINAL PARA TU TFG\n');
fprintf('======================================================\n');
T = cell2table(resumen_final, 'VariableNames', {'Archivo', 'Split', 'MAE_m', 'P90_m', 'Tiempo_Seg', 'Configuracion'});
disp(T);
fprintf('️ TIEMPO TOTAL DE PROCESO: %.2f minutos\n', toc(tic_global)/60);
fprintf('======================================================\n');