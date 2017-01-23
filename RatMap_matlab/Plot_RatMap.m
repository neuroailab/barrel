% Plot_RatMap
% Brian Quist
% March 4, 2011

%% + Generate RatMap
[x,y,z] = Get_RatMap();
[ER,EL] = Get_RatMapEllipsoid();

%% + Plot: Ellipsoids
hold on;
hR = surf(ER.x,ER.y,ER.z); shading interp; hold on;
hL = surf(EL.x,EL.y,EL.z); shading interp; hold on;
set(hR,'FaceAlpha',0.05,'FaceColor','k');
set(hL,'FaceAlpha',0.05,'FaceColor','k');

%% + Plot: Whiskers
clr = jet(size(x,1));
for ii = 1:size(x,1)
   plot3(x(ii,:),y(ii,:),z(ii,:),'.-','Color',clr(ii,:)); hold on;
end

%% + Format
xlabel('x');
ylabel('y');
zlabel('z');
axis equal; 
grid on;
view(150,20);