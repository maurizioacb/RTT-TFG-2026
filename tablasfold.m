% =========================================================================
% script que hace las tablas
% =========================================================================
clear; clc;
carpeta = 'C:\Users\mauac\OneDrive\Documentos\TFG\Datasets and Methods\databases\';
archivos = dir(fullfile(carpeta, '*.mat'));
dataset_completo = struct();

mapa_puntos = [
    2.65, 13.36, 1; 2.77, 0.50, 1; 5.14, 6.68, 1; 5.20, 13.33, 1;
    2.77, 10.07, 2; 5.19, 10.03, 2; 5.19, 11.70, 2; 6.32, 1.89, 2;
    0.92, 13.35, 3; 1.19, 0.52, 3; 2.66, 6.68, 3; 6.32, 5.70, 3;
    1.22, 6.70, 4; 1.28, 3.46, 4; 5.24, 0.74, 4; 6.32, 13.48, 4;
    1.14, 10.07, 5; 2.67, 3.47, 5; 5.17, 8.35, 5; 5.22, 3.45, 5
];

folds_data = cell(5, 1);
for f = 1:5, folds_data{f} = mapa_puntos(mapa_puntos(:,3) == f, 1:2); end

for a = 1:length(archivos)
    nombre = archivos(a).name;
    fprintf('Procesando: %s\n', nombre);
    db = load(fullfile(carpeta, nombre)).database;
    
    for i = 1:5
        folds_test = i; folds_train = setdiff(1:5, i);
        s.entreno.trainingMacs = filtrar_tol(db.trainingMacs, db.trainingLabels, folds_data, folds_train);
        s.entreno.trainingLabels = filtrar_tol(db.trainingLabels, db.trainingLabels, folds_data, folds_train);
        s.test.testMacs = filtrar_tol(db.testMacs, db.testLabels, folds_data, folds_test);
        s.test.testLabels = filtrar_tol(db.testLabels, db.testLabels, folds_data, folds_test);
        dataset_completo.(strrep(nombre, '.mat', '')).splits{i} = s;
    end
end
fprintf('Fábrica completada. Variable "dataset_completo" lista.\n');

function out = filtrar_tol(data, labels, folds, idxs)
    pts = []; for i = idxs, pts = [pts; folds{i}]; end
    idx = false(size(labels, 1), 1);
    for p = 1:size(pts, 1)
        dist = sqrt((labels(:,1)-pts(p,1)).^2 + (labels(:,2)-pts(p,2)).^2);
        idx = idx | (dist < 0.05);
    end
    out = data(idx, :);
end