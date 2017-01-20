function [ER,EL] = Get_RatMapEllipsoid(varargin)
% function [ER,EL] = Get_RatMapEllipsoid({setting_name,setting})
% -------------------------------------------------------------------------
% INPUT:
%   setting_name - optional setting (see SETTINGS section)
%   setting - corresponding setting for setting_name (see SETTINGS section)
% OUTPUT:
%   ER - Right-side ellipsoid struct of 3D points (.x, .y, .z)
%   EL - Left-side ellipsoid struct of 3D points (.x, .y, .z)
% ------------------------------------------------------------------------
% EXAMPLE FUNCTION CALLS:
%   [ER,EL] = Get_RatMapEllipsoid();      
%                   Returns both L&R complete ellipsoids
%   [ER,EL] = Get_RatMapEllipsoid('E_C',[9 9 9]);      
%                   Returns both L&R complete ellipsoids with new ellipsoid
%                   center for 'R' ellipsoid, that is reflected across
%                   zy-plane to get 'L' ellipsoid
% -------------------------------------------------------------------------
% SETTINGS:
%   'E_C'       - ellipsoid centroid ([# # # mm]) where ([x y z])
%   'E_R'       - ellipsoid radii ([# # # mm]) where ([ra rb rc])
%   'E_OA'      - ellipsoid orientation angles ([# # # degrees])
%                   where rotation is about ([z y x]) axes 
% + Spherical coordinates used for calculation:
%   'TH_range'  - [th_start th_inc th_end] ([# # # degrees])
%   'PHI_range' - [phi_start phi_inc phi_end] ([# # # degrees])
% -------------------------------------------------------------------------
% NOTES:
% + ellipsoid parameters set relative to 'R' ellipsoid, 
%       then mirrored for 'L'
% + Portions of this code were adapted from the work of Blythe Towal
% + About the calculations:
%   * Ellipsoid: Computed for *right* side, then reflected across zy-plane
%       to get left-side
% -------------------------------------------------------------------------
% Brian Quist
% March 4, 2011
ER = NaN; EL = NaN; %#ok<NASGU>

%% + Setup defaults and process settings
% Defaults
S = LOCAL_SetupDefaults();
% Process settings
if nargin >= 2,
    settings = varargin;
    S = LOCAL_ProcessSettings(S,settings);
end

%% + Construct RIGHT ellipsoid
ER = LOCAL_ConstructEllipsoid(S);

%% + Construct LEFT ellipsoid
EL = ER; 
EL.x = EL.x.*(-1);

function S = LOCAL_SetupDefaults()
%% function S = LOCAL_SetupDefaults()

% Setup ellipsoid defaults (for 'R' ellipsoid)
S.E_C = [1.9128 -7.6549 -5.4439];
S.E_R = [9.5304 5.5393 6.9745];
S.E_OA = [106.5100 -2.5211 -19.5401];

% Calculation defaults
S.TH_range = [0 5 360];
S.PHI_range = [0 5 360];

function S = LOCAL_ProcessSettings(S,settings)
%% function S = LOCAL_ProcessSettings(S,settings)
for ii = 1:2:length(settings)
    switch settings{ii}
        case 'E_C',         S.E_C = settings{ii+1};
        case 'E_R',         S.E_R = settings{ii+1};
        case 'E_OA',        S.E_OA = settings{ii+1};
        case 'TH_range',    S.TH_range = settings{ii+1};
        case 'PHI_range',   S.PHI_range = settings{ii+1};            
        otherwise
            error(['ERROR: ',settings{ii},' is not a valid setting']);
    end  
end

function E = LOCAL_ConstructEllipsoid(S)
%% E = LOCAL_ConstructEllipsoid(S)

% Convert to radians
S.TH_range = S.TH_range.*(pi/180);
S.PHI_range = S.PHI_range.*(pi/180);

% Construct grid
[U,V] = meshgrid( ...
    S.TH_range(1):S.TH_range(2):S.TH_range(3), ...
    S.PHI_range(1):S.PHI_range(2):S.PHI_range(3));

% Compute the ellipsoid
xx = S.E_R(1).*cos(U).*cos(V);
yy = S.E_R(2)*cos(U).*sin(V);
zz = S.E_R(3)*sin(U);

% Construct ellipsoid rotation matrix
% (see calculation in LOCAL_ConstructWhisker3D in Get_RatMap)
c_x = cos(S.E_OA(3)*(pi/180));
s_x = sin(S.E_OA(3)*(pi/180));
c_y = cos(S.E_OA(2)*(pi/180));
s_y = sin(S.E_OA(2)*(pi/180));
c_z = cos(S.E_OA(1)*(pi/180));
s_z = sin(S.E_OA(1)*(pi/180));
A = [...
    c_y*c_z, c_z*s_x*s_y - c_x*s_z, s_x*s_z + c_x*c_z*s_y; ...
    c_y*s_z, c_x*c_z + s_x*s_y*s_z, c_x*s_y*s_z - c_z*s_x; ...
    -s_y,    c_y*s_x,               c_x*c_y];

% Rotate
E.x = xx.*A(1,1)+yy.*A(1,2)+zz.*A(1,3);
E.y = xx.*A(2,1)+yy.*A(2,2)+zz.*A(2,3);
E.z = xx.*A(3,1)+yy.*A(3,2)+zz.*A(3,3);

% Translate
EYE = ones(size(xx));
E.x = E.x + EYE.*S.E_C(1);
E.y = E.y + EYE.*S.E_C(2);
E.z = E.z + EYE.*S.E_C(3);