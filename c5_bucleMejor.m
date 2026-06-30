% =========================================================================
% k-nn configuracion 5
% =========================================================================

clear; clc;
addpath('metodos'); 

archivos = dir('databases/*R*.mat'); 

metodos_list = {'method_knn_plainrtt2025', 'method_wknnid_plainrtt2025', 'method_wknnsid_plainrtt2025'};
metricas = {'distancem_cityblock', 'distance_euclidean', 'distancem_minkowsky3', ...
            'distancem_cosine', 'distancem_sorensen', 'distancem_neyman'}; 
k_list = 1:2:51; 

nombres_id = {"POCO Persona", "POCO Trípode", "S24U Persona", "S24U Trípode"};

errores_finales = [];
errores_tr_AVE = [];
errores_tr_RAW = [];
errores_tr_W1S = [];

fprintf('Iniciando Optimización Dinámica de Vecinos Cercanos (Estrategia C5)...\n\n');

for f = 1:length(archivos)
    nombre = archivos(f).name;
    partes = split(nombre, '_');
    
    if length(partes) < 4, continue; end
    if ~strcmp(partes{3}, 'RAW'), continue; end
    
    ids = regexp(nombre, '\d+', 'match');
    if length(ids) < 1, continue; end
    id_pair = ids{end}; 
    
    if length(id_pair) >= 2
        id_tr = str2double(id_pair(1));
        id_te = str2double(id_pair(2));
    else
        continue;
    end
    
    if id_tr == id_te || id_tr == 0 || id_te == 0, continue; end
    
    try
        data_struct = load(fullfile('databases', nombre)); 
        f_names = fieldnames(data_struct);
        db = data_struct.(f_names{1}); 
        
        minTr = min(db.trainingMacs, [], 2, 'omitnan');
        minTe = min(db.testMacs, [], 2, 'omitnan');
        db.trainingMacs = db.trainingMacs - minTr;
        db.testMacs = db.testMacs - minTe;
    catch
        continue;
    end
    
    mejor_err = inf;
    mejor_config = '';
    
    for f_idx = 1:length(metodos_list)
        func_name = metodos_list{f_idx};
        for m = 1:length(metricas)
            for k_val = k_list
                try
                    ejecucion = sprintf('[mudo, res] = evalc(''%s(db, %d, ''''rtt'''', NaN, 100000, ''''%s'''')'');', ...
                                func_name, k_val, metricas{m});
                    eval(ejecucion);
                    
                    err = mean(res.error(:,1), 'omitnan');
                    if err < mejor_err
                        mejor_err = err;
                        tipo_voto = strrep(func_name, '_plainrtt2025', '');
                        tipo_voto = strrep(tipo_voto, 'method_', '');
                        mejor_config = sprintf('%s, K=%d, %s', upper(tipo_voto), k_val, strrep(metricas{m}, 'distancem_', ''));
                    end
                catch
                    continue; 
                end
            end
        end
    end
    
    if mejor_err < inf
        agg_tr = partes{2};
        desc = sprintf('%s (Train: %s %s | Test: W1S %s)', ...
               nombre, agg_tr, nombres_id{id_tr}, nombres_id{id_te});
        
        fprintf('%-85s | %-25s | Error: %.4f m\n', desc, mejor_config, mejor_err);
        
        errores_finales = [errores_finales; mejor_err];
        switch agg_tr
            case 'AVE', errores_tr_AVE = [errores_tr_AVE; mejor_err];
            case 'RAW', errores_tr_RAW = [errores_tr_RAW; mejor_err];
            case 'W1S', errores_tr_W1S = [errores_tr_W1S; mejor_err];
        end
    end
end

fprintf('\n--- RESUMEN DE PROMEDIOS GLOBALES OPTIMIZADOS (KNN C5) --- \n');
if ~isempty(errores_tr_AVE), fprintf('Training: AVE  | Media Error: %.4f m\n', mean(errores_tr_AVE)); end
if ~isempty(errores_tr_RAW), fprintf('Training: RAW  | Media Error: %.4f m\n', mean(errores_tr_RAW)); end
if ~isempty(errores_tr_W1S), fprintf('Training: W1S  | Media Error: %.4f m\n', mean(errores_tr_W1S)); end
fprintf('PRECISIÓN GLOBAL OPTIMIZADA: %.4f metros\n', mean(errores_finales));
beep;