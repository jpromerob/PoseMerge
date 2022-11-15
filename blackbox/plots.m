clear; clc; close all


hl_array = 4:2:16;
nb_nn = length(hl_array);

px_e = [0,4,8,12,16,20,24,32];
nb_ds = length(px_e);

cc_array = 1:1:20;
nb_copies = length(cc_array);


error = load('BikeErrorAll.mat').error;

mean_e = mean(error,3);
sdev_e = std(error,1,3);

fig_px = figure;
for hl_ix=1:nb_nn
    plot(px_e, mean_e(:,hl_ix), 'DisplayName', sprintf('HN: %d', hl_array(hl_ix)))
    hold on;
end
xlabel('Mean Noise in Px')
ylabel('Error in [mm]')
lgd = legend('Location','northwest');
title(lgd,'# Hidden Units')
lgd.NumColumns = 2;
xlim([min(px_e), max(px_e)])
grid on;
saveas(fig_px, 'fig_px.png')

fig_hn = figure;
for ds_ix=1:nb_ds
    plot(hl_array, mean_e(ds_ix,:), 'DisplayName', sprintf('%d px', px_e(ds_ix)));
    hold on;
end
xlabel('# Hidden Units')
ylabel('Error in [mm]')
lgd = legend('Location','northeast');
title(lgd,'Mean Noise in Px')
lgd.NumColumns = 2;
grid on;
saveas(fig_hn, 'fig_hn.png')

fig_box = figure;
for ds_ix=1:nb_ds
    aux = transpose(squeeze(error(ds_ix,:,:)));
    boxplot(aux,'Notch','on','Labels',mat2cell(hl_array,1));
    grid on;
    xlabel('# Hidden Units');
    ylabel('Error in [mm]');
    ylim([0,32]);
    title(sprintf("Results for dataset with %d px noise",px_e(ds_ix)));
    saveas(fig_box, sprintf("box_%dpx.png", px_e(ds_ix)));
    pause(0.01)
end

for hl_ix=1:nb_nn
    aux = transpose(squeeze(error(:,hl_ix,:)));
    boxplot(aux,'Notch','on','Labels',mat2cell(px_e,1));
    grid on;
    xlabel('Means Noise in Px');
    ylabel('Error in [mm]');
    ylim([0,32]);
    title(sprintf("Results for Network with %d HN",hl_array(hl_ix)));
    saveas(fig_box, sprintf("box_%dhn.png", hl_array(hl_ix)));
    pause(0.01)
end
    

