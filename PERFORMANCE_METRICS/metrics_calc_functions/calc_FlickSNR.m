function [FSNR,F_sig,F_bg] = calc_FlickSNR(Signal_events,BG_events,matrix_size)

init_mat = zeros(matrix_size(1),matrix_size(2));
F_sig = init_mat;
F_bg = init_mat;

for xi =1:matrix_size(1)
    for yi=1:matrix_size(2)
        ind = (Signal_events.x==xi) & (Signal_events.y==yi);
        if any(ind)
            on = Signal_events.on(find(ind));
            F_sig(xi,yi) = 2*sum(on)*sum(1-on)/(length(on));
        end
        ind = (BG_events.x==xi) & (BG_events.y==yi);
        if any(ind)
            on = BG_events.on(find(ind));        
            F_bg(xi,yi) = 2*sum(on)*sum(1-on)/(length(on));
        end
    end
end

FSNR = max(F_sig(:))/std(F_bg(:));

        