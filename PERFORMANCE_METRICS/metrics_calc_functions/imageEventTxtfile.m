function imageEventTxtfile(filepath)
eventdata = load(filepath);
events.x = eventdata(:,1)+1;
events.y = eventdata(:,2)+1;
events.on = eventdata(:,3);
events.t = eventdata(:,4);
events.t = events.t - events.t(1) + mod(events.t(1),100);
events.label = eventdata(:,5);
sig_ind = events.label<0;

figure;
plot3(events.x(sig_ind),events.y(sig_ind),events.t(sig_ind),'r.','MarkerSize',0.5);
hold on;
plot3(events.x(~sig_ind),events.y(~sig_ind),events.t(~sig_ind),'b.','MarkerSize',0.5);
grid on;
xlabel('X')
ylabel('Y')
zlabel('t')