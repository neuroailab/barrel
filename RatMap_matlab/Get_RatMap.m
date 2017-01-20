function [x,y,z,varargout] = Get_RatMap(varargin)
%% function [x,y,z,{S}] = Get_RatMap({wselect,setting_name,setting, ...})
% -------------------------------------------------------------------------
% INPUT:
%   wselect - select whiskers of interest 
%           Use 'R'-'L' for Right or Left.
%               '0'-'6' for columns (0) Greek to (6) Column 6
%               'A'-'E' for Rows A to E
%               (note: possible ranges are A0-A4,B0-B5,C0-C6,D0-D6,E1-E6)
%           e.g. 'RA1': Right A1 whisker
%                'A1' : Both Right/Left A1 whisker
%                '1'  : Both Right/Left Column 1 whiskers
%                'A'  : Both Right/Left Row A whiskers
%                ''   : Both Right/Left ALL whiskers
%   setting_name - optional setting (see SETTINGS section)
%   setting - corresponding setting for setting_name (see SETTINGS section)
% OUTPUT:
%   x - x-coordinates of whisker(s)
%   y - y-coordinates of whisker(s)
%   z - z-coordinates of whisker(s)
%   varargout:
%   {1}: S - struct of set/computed parameters
%           (see S-STRUCT CONTENTS section below for more details)
% ------------------------------------------------------------------------
% EXAMPLE FUNCTION CALLS:
%   [x,y,z] = Get_RatMap();      Returns both L&R full arrays
%   [x,y,z] = Get_RatMap('L');   Returns L full array
%   [x,y,z] = Get_RatMap('LA');  Returns L A-row
%   [x,y,z] = Get_RatMap('LA1'); Returns LA1 whisker
%   [x,y,z] = Get_RatMap('','Npts',50); 
%                                Returns full array where whiskers
%                                have 50 data points each
%   [x,y,z] = Get_RatMap('L','EQ_W_th',[9.9 9.9 99.9]);
%                               Returns L full array with new function for
%                               theta that depends on row and col
%   [x,y,z] = Get_RatMap('LA1','W_th',90); 
%                               Returns LA1 whisker with a projection
%                               theta set to 90 degrees
% -------------------------------------------------------------------------
% SETTINGS:
%   'Npts'      - Number of points ([#])
%   'TGL_PHI'   - Select method to compute euler PHI angle:
%                   Toggle using either: 
%                   'proj' - {default} - Uses projection angles phi and psi 
%                   'proj_phi_only' - Only uses projection angle phi                     
%                   'euler' - Uses euler phi
%   'E_C'       - ellipsoid centroid ([# # # mm]) where ([x y z])
%   'E_R'       - ellipsoid radii ([# # # mm]) where ([ra rb rc])
%   'E_OA'      - ellipsoid orientation angles ([# # # degrees])
%                   where rotation is about ([z y x]) axes 
%
% + If changing the parameters below, either change the equation('EQ_') 
%   or the parameter. If the parameter is changed, then it will be used 
%   instead of the equation. 
%
%   'EQ_BP_th'  - base point theta equation ([# # #])
%   'BP_th'     - base-point theta ([# degrees])
%
%   'EQ_BP_phi' - base point phi equation ([# # #])
%   'BP_phi'    - base-point phi ([# degrees])
%
%   'EQ_W_s'    - whisker length equation ([# # #])
%   'W_s'       - whisker arc length ([# mm])
%
%   'EQ_W_a'    - whisker quadratic coefficient equation ([# # #])
%   'W_a'       - whisker quadratic coefficient ([# 1/mm])
%
%   'EQ_W_th'   - whisker theta equation ([# # #])
%   'W_th'      - whisker theta ([# degrees])
%
%   'EQ_W_psi'  - whisker psi equation ([# # #])
%   'W_psi'     - whisker psi ([# degrees])
%
%   'EQ_W_zeta' - whisker zeta equation ([# # #])
%   'W_zeta'    - whisker zeta ([# degrees])
%
% + The phi angle that is used (projection phi or euler phiE) will depend
%   on the 'TGL_PHI' set above.
%
%   'EQ_W_phi'  - whisker phi equation ([# # #])
%   'W_phi'     - whisker phi ([# degrees])
%   'EQ_W_phiE' - whisker euler phi equation ([# # #])
%   'W_phiE'    - whisker euler phi ([# degrees])
%
% -------------------------------------------------------------------------
% S-STRUCT CONTENTS (computed within Get_RatMap):
%
% + In addition to the parameters listed in 'SETTINGS', the S-struct
%   contains values directly used to compute the 3D whisker location:
%
%   'wname'     - list of whisker names computed (row index of each whisker 
%                   name corresponds to row of each computed parameter)
%   'C_BP_th'   - calculated base-point theta (degrees)
%   'C_BP_phi'  - calculated base-point phi (degrees)
%   'C_s'       - calculated whisker arc length (mm)
%   'C_a'       - calculated whisker quadratic coefficient (1/mm)
%   'C_thetaP'  - calculated projection angle theta (degrees)
%   'C_phiP'    - calculated projection angle phi (degrees) 
%   'C_psiP'    - calculated projection angle psi (degrees)
%   'C_zetaP'   - calculated projection angle zeta (degrees)
%
% + Final values used to rotate and translate the whisker:
%
%   'C_zeta'    - calculated zeta euler rotation angle (radians)
%   'C_theta'   - calculated theta euler rotation angle (radians)
%   'C_phi'     - calculated phi euler rotation angle (radians)
%   'C_baseX'   - calculated base-point x-location (mm)
%   'C_baseY'   - calculated base-point y-location (mm)
%   'C_baseZ'   - calculated base-point z-location (mm)
%    
% -------------------------------------------------------------------------
% NOTES:
%
% + When using this function, users can compute array parameters by:
%       1. Equation form (either the default or by inputting the equation)
%       2. Directly specifying the parameter
%           a. Single number provided: this value will be used for entire
%               array (e.g. ... ,'W_s',10,... will make all whiskers in the
%               array have a length of 10 mm)
%           b. Vector of numbers provided: Assumes the size of the vector
%               provided matches the number of whiskers specified 
%               with 'wselect' 
%       * If both an equation and a value are provided, then the new value 
%           will be used instead of the equation. 
%
% + Settings with 'EQ' label require a vector with three inputs that 
%   correspond to the array matrix location, so that equation coefficients:
%
%       EQ_X = [ (COL coefficient) (ROW coeffiecient) (Offset)]
%
%   are used to compute each whisker's parameter (X) using the equation:
%
%       X = EQ_X(1).*COL + EQ_X(2).*ROW + EQ_X(3);
%
% + Whisker euler PHI orientation can be computed by one of three methods:
%       1. Conversion of projection angles (PHI,PSI) to euler angle PHI
%           (projection angles match those shown in the publication)
%           -> set 'TGL_PHI' to 'proj'  {default}   
%       2. Conversion of projection angle (PHI) to euler angle PHI
%           (uses PHI for complete range of projection thetas)
%           -> set 'TGL_PHI' to 'proj_phi_only'
%       3. Direct use of euler PHI angle
%           (either using equation from regression fit to data in
%           publication, user-specified equation, or user-specified value)
%           -> set 'TGL_PHI' to 'euler'
%
% + When adding motion (i.e. arbitrarily changing theta and/or phi),
%   the 'TGL_PHI' should be set to 'euler'. This is because the angles
%   thetaP, phiP and psiP are numerically related and should not be changed
%   independently.
%   
%   For example, a point in 3D space (x0,y0,z0) with (x0 > 0), (y0 > 0), 
%   and (z0 > 0) has the associated angles:
%       tan(theta) = y0/x0        tan(phi) = z0/x0      tan(psi) = z0/y0
%   where these angles are related by the equation:
%       tan(phi) = tan(theta)*tan(psi)
%
%   Simply changing one projection angle while keeping the others constant
%   breaks the functional relationship between the three angles. 
%
% + Greek whiskers have been associated with a row so that:
%   alpha = A0; beta = B0; gamma = C0; delta = D0;
%
% + About the calculations:
%   * Ellipsoid: Computed for *right* side, then reflected across zy-plane 
%       to get left-side
%   * Base-points: Computed for *right* side, then reflected across 
%       zy-plane to get left-side points
%
% + Portions of this code were adapted from the work of Blythe Towal
% -------------------------------------------------------------------------
% Brian Quist
% March 4, 2011
x = NaN; y = NaN; z = NaN; %#ok<NASGU>

%% + Setup defaults and process settings
% Defaults
S = LOCAL_SetupDefaults();
% Setup possible whisker names
wselect = [];
if nargin >= 1 && ~isempty(varargin{1}), wselect = varargin{1}; end
S.wname = LOCAL_SetupWhiskerNames(wselect);
% Process settings
if nargin >= 3,
    settings = varargin(2:end);
    S = LOCAL_ProcessSettings(S,settings);
end
S = LOCAL_UnpackEquations(S);

%% + Calculate parameters for each whisker
S = LOCAL_CalculateParameters(S);

%% + Calculate base-point 3D locations
S = LOCAL_Calculate3DBasePoints(S);

%% + Construct whisker
[x,y,z] = LOCAL_ConstructWhisker3D(S);
varargout{1} = S;

function S = LOCAL_SetupDefaults()
%% function S = LOCAL_SetupDefaults()

% Calcualtion defaults
S.Npts = 100;
S.TGL_PHI = 'proj';

% Setup ellipsoid defaults
S.E_C = [1.9128 -7.6549 -5.4439];
S.E_R = [9.5304 5.5393 6.9745];
S.E_OA = [106.5100 -2.5211 -19.5401];

% Setup whisker transformation parameters and equations
% (parameters must be NaN to use equations)

S.EQ_BP_th = [15.2953 0 -144.2220];
S.BP_th = NaN;

S.EQ_BP_phi = [0 18.2237 34.7558];
S.BP_phi = NaN;

S.EQ_W_s = [-7.9312 2.2224 52.1110];
S.W_s = NaN;

S.EQ_W_a = [-0.02052 0 -0.2045];
S.W_a = NaN;

S.EQ_W_th = [10.6475 0 37.3178];
S.W_th = NaN;

S.EQ_W_psi = [18.5149 49.3499 -50.5406];
S.W_psi = NaN;

S.EQ_W_zeta = [18.7700 -11.3485 -4.9844];
S.W_zeta = NaN;

S.EQ_W_phi = [1.0988 -18.0334 50.6005];
S.W_phi = NaN;
S.EQ_W_phiE = [0 -15.8761 47.3263];
S.W_phiE = NaN;

function wname = LOCAL_SetupWhiskerNames(wselect)
%% function wname = LOCAL_SetupWhiskerNames(wselect)

% Setup defaults
wname = ['A0';'A1';'A2';'A3';'A4'; ...
          'B0';'B1';'B2';'B3';'B4';'B5'; ...
          'C0';'C1';'C2';'C3';'C4';'C5';'C6'; ...
          'D0';'D1';'D2';'D3';'D4';'D5';'D6'; ...
          'E1';'E2';'E3';'E4';'E5';'E6'];
wname = [ ...
    [repmat('L',size(wname,1),1) wname];  ...
    [repmat('R',size(wname,1),1) wname]];

% Select whiskers
for ii = 1:length(wselect)
    if ~isempty(wname),
        if strcmp(wselect(ii),'R') || strcmp(wselect(ii),'L'),
            % Right/Left
            wname = wname(strmatch(wselect(ii),wname(:,1)),:);
        elseif strcmp(wselect(ii),'A') || strcmp(wselect(ii),'B') || ...
                strcmp(wselect(ii),'C') || strcmp(wselect(ii),'D') || ...
                strcmp(wselect(ii),'E'),
            % Row A-E
            wname = wname(strmatch(wselect(ii),wname(:,2)),:);
        elseif strcmp(wselect(ii),'0') || strcmp(wselect(ii),'1') || ...
                strcmp(wselect(ii),'2') || strcmp(wselect(ii),'3') || ...
                strcmp(wselect(ii),'4') || strcmp(wselect(ii),'5') || ...
                strcmp(wselect(ii),'6'),
            % Columns 0-6
            wname = wname(strmatch(wselect(ii),wname(:,3)),:);
        else
            error('ERROR: Invalid character in wselect');
        end
    else
        error('ERROR: ''wselect'' not set properly. No whisker selected');
    end
end

function S = LOCAL_ProcessSettings(S,settings)
%% function S = LOCAL_ProcessSettings(S,settings)
for ii = 1:2:length(settings)
    switch settings{ii}
        case 'Npts',        S.Npts = settings{ii+1};
        case 'TGL_PHI',     S.TGL_PHI = settings{ii+1};
        case 'BP_th',       S.BP_th = settings{ii+1};
        case 'BP_phi',      S.BP_phi = settings{ii+1};
        case 'W_s',         S.W_s = settings{ii+1};
        case 'W_a',         S.W_a = settings{ii+1};
        case 'W_th',        S.W_th = settings{ii+1};
        case 'W_phi',       S.W_phi = settings{ii+1};
        case 'W_phiE',      S.W_phiE = settings{ii+1};
        case 'W_psi',       S.W_psi = settings{ii+1};
        case 'W_zeta',      S.W_zeta = settings{ii+1};
        case 'EQ_BP_th',    S.EQ_BP_th = settings{ii+1};
        case 'EQ_BP_phi',   S.EQ_BP_phi = settings{ii+1};
        case 'EQ_W_s',      S.EQ_W_s = settings{ii+1};
        case 'EQ_W_a',      S.EQ_W_a = settings{ii+1};
        case 'EQ_W_th',     S.EQ_W_th = settings{ii+1};
        case 'EQ_W_phi',    S.EQ_W_phi = settings{ii+1};
        case 'EQ_W_phiE',   S.EQ_W_phiE = settings{ii+1};
        case 'EQ_W_psi',    S.EQ_W_psi = settings{ii+1};
        case 'EQ_W_zeta',   S.EQ_W_zeta = settings{ii+1};
        case 'E_C',         S.E_C = settings{ii+1};
        case 'E_R',         S.E_R = settings{ii+1};
        case 'E_OA',        S.E_OA = settings{ii+1};
        otherwise
            error(['ERROR: ',settings{ii},' is not a valid setting']);
    end  
end

function S = LOCAL_UnpackEquations(S)
%% function S = LOCAL_UnpackEquations(S)

% Setup logical toggles ---------------------------------------------------
ROW = zeros(size(S.wname,1),1);
COL = zeros(size(S.wname,1),1);
ltr = 'ABCDE'; nm = '0123456';
for ii = 1:7
    if ii <= 5, 
        ROW(logical(S.wname(:,2) == ltr(ii))) = ii;
    end
    COL(logical(S.wname(:,3) == nm(ii))) = ii;
end
EYE = ones(size(S.wname,1),1);

% Base-points -------------------------------------------------------------

% (BP_th) BASE-POINT THETA
if ~isnan(S.BP_th),
    S.C_BP_th = [S.BP_th; S.BP_th];
else
    S.C_BP_th = S.EQ_BP_th(1).*COL + S.EQ_BP_th(2).*ROW + S.EQ_BP_th(3);
end

% (BP_phi) BASE-POINT PHI
if ~isnan(S.BP_phi),
    S.C_BP_phi = [S.BP_phi; S.BP_phi];
else
    S.C_BP_phi = S.EQ_BP_phi(1).*COL + S.EQ_BP_phi(2).*ROW + S.EQ_BP_phi(3);
end

% 2D shape ----------------------------------------------------------------

% (s) ARCLENGTH
if ~isnan(S.W_s(1)) && max(size(S.W_s)) == 1,
    S.C_s = EYE.*S.W_s;
elseif ~isnan(S.W_s(1)) && max(size(S.W_s)) ~= 1,
    if max(size(S.W_s)) ~= size(S.wname,1),
        error('size(S.W_s) does not match the number of whiskers');
    end
    S.C_s = S.W_s;
else
    S.C_s = S.EQ_W_s(1).*COL + S.EQ_W_s(2).*ROW + S.EQ_W_s(3);
end

% (a) QUADRATIC COEFFICIENT
if ~isnan(S.W_a(1)) && max(size(S.W_a)) == 1,
    S.C_a = EYE.*S.W_a;
elseif ~isnan(S.W_a(1)) && max(size(S.W_a)) ~= 1,
    if max(size(S.W_a)) ~= size(S.wname,1),
        error('size(S.W_a) does not match the number of whiskers');
    end
    S.C_a = S.W_a;
else
    S.C_a = exp(1./(S.EQ_W_a(1).*COL + S.EQ_W_a(2).*ROW + S.EQ_W_a(3)));
end

% PROJECTION ANGLES -------------------------------------------------------

% Theta
if ~isnan(S.W_th(1)) && max(size(S.W_th)) == 1,
    S.C_thetaP = EYE.*S.W_th;
elseif ~isnan(S.W_th(1)) && max(size(S.W_th)) ~= 1,
    if max(size(S.W_th)) ~= size(S.wname,1),
        error('size(S.W_th) does not match number of whiskers');
    end
    S.C_thetaP = S.W_th;
else
    S.C_thetaP = S.EQ_W_th(1).*COL + S.EQ_W_th(2).*ROW + S.EQ_W_th(3);
end

% Phi
if ~isnan(S.W_phi(1)) && max(size(S.W_phi)) == 1,
    S.C_phiP = EYE.*S.W_phi;
elseif ~isnan(S.W_phi(1)) && max(size(S.W_phi)) ~= 1,
    if max(size(S.W_phi)) ~= size(S.wname,1),
        error('size(S.W_phi) does not match the number of whiskers');
    end
    S.C_phiP = S.W_phi;
else
    S.C_phiP = S.EQ_W_phi(1).*COL + S.EQ_W_phi(2).*ROW + S.EQ_W_phi(3);
end

% Psi
if ~isnan(S.W_psi(1)) && max(size(S.W_psi)) == 1,
    S.C_psiP = EYE.*S.W_psi;
elseif ~isnan(S.W_psi(1)) && max(size(S.W_psi)) ~= 1,
    if max(size(S.W_psi)) ~= size(S.wname,1),
        error('size(S.W_psi) does not match the number of whiskers');
    end
    S.C_psiP = S.W_psi;
else
    S.C_psiP = S.EQ_W_psi(1).*COL + S.EQ_W_psi(2).*ROW + S.EQ_W_psi(3);
end

% Zeta
if ~isnan(S.W_zeta(1)) && max(size(S.W_zeta)) == 1,
    S.C_zetaP = EYE.*S.W_zeta;
elseif ~isnan(S.W_zeta(1)) && max(size(S.W_zeta)) ~= 1,
    if max(size(S.W_zeta)) ~= size(S.wname,1),
        error('size(S.W_zeta) does not match the number of whiskers');
    end    
    S.C_zetaP = S.W_zeta;
else
    S.C_zetaP = S.EQ_W_zeta(1).*COL + S.EQ_W_zeta(2).*ROW + S.EQ_W_zeta(3);
end

% EULER ANGLES -------------------------------------------------------

% PhiE
if ~isnan(S.W_phiE(1)) && max(size(S.W_phiE)) == 1,
    S.C_phiE = EYE.*S.W_phiE;
elseif ~isnan(S.W_phiE(1)) && max(size(S.W_phiE)) ~= 1,
    if max(size(S.W_phiE)) ~= size(S.wname,1),
        error('size(S.W_phiE) does not match the number of whiskers');
    end    
    S.C_phiE = S.W_phiE;
else
    S.C_phiE = S.EQ_W_phiE(1).*COL + S.EQ_W_phiE(2).*ROW + S.EQ_W_phiE(3);
end

function S = LOCAL_CalculateParameters(S)
%% function S = LOCAL_CalculateParameters(S)

d2r = pi/180;
SIDE = logical(S.wname(:,1) == 'R');

% EULER ZETA --------------------------------------------------------------
S.C_zeta = zeros(size(S.C_zetaP));
% If 'R', 
S.C_zeta(SIDE) = S.C_zetaP(SIDE)+90;
% If 'L',
S.C_zeta(~SIDE) = 90-S.C_zetaP(~SIDE);

% EULER THETA -------------------------------------------------------------
S.C_theta = zeros(size(S.C_thetaP));
% Segment theta:
Q1 = logical(90 <= S.C_thetaP & S.C_thetaP <= 200 & SIDE); % CHANGED from 180
Q2 = logical( 0 <= S.C_thetaP & S.C_thetaP <  90  & SIDE);
Q3 = logical(90 <= S.C_thetaP & S.C_thetaP <= 200 & ~SIDE); % CHANGED from 180
Q4 = logical( 0 <= S.C_thetaP & S.C_thetaP <  90  & ~SIDE);
% Compute
S.C_theta(Q1) = S.C_thetaP(Q1)-90;
S.C_theta(Q2) = S.C_thetaP(Q2)+270;
S.C_theta(Q3) = (180-S.C_thetaP(Q3))+90;
S.C_theta(Q4) = 270-S.C_thetaP(Q4);
% Side effect: already taken into account

% EULER PHI ---------------------------------------------------------------
S.C_phi = zeros(size(S.C_phiP));
switch S.TGL_PHI
    case 'euler',
        S.C_phi = S.C_phiE.*(d2r);
        
    case 'proj',
        % Segment
        Q1 = logical(  0 <= S.C_theta & S.C_theta <=  45);
        Q2 = logical( 45 <  S.C_theta & S.C_theta <=  90);
        Q3 = logical( 90 <  S.C_theta & S.C_theta <= 135);
        Q4 = logical(135 <  S.C_theta & S.C_theta <= 225);
        Q5 = logical(225 <  S.C_theta & S.C_theta <= 270 & 0 <= S.C_psiP & S.C_psiP <= 90);
        Q6 = logical(225 <  S.C_theta & S.C_theta <= 270 & 270 <= S.C_psiP);
        Q7 = logical(270 <  S.C_theta & S.C_theta <= 315 & 0 <= S.C_psiP & S.C_psiP <= 90);
        Q8 = logical(270 <  S.C_theta & S.C_theta <= 315 & 270 <= S.C_psiP);
        Q9 = logical(315 <  S.C_theta & S.C_theta <= 360);
        % Compute
        S.C_phi(Q1) = atan(tan(S.C_phiP(Q1).*d2r) ...
            .*cos(S.C_theta(Q1).*d2r));
        S.C_phi(Q2) = atan(tan((180-S.C_psiP(Q2)).*d2r) ...
            .*sin(S.C_theta(Q2).*d2r));
        S.C_phi(Q3) = atan(tan((180-S.C_psiP(Q3)).*d2r) ...
            .*sin((180-S.C_theta(Q3)).*d2r));        
        S.C_phi(Q4) = atan(tan(S.C_phiP(Q4).*d2r) ...
            .*cos((180-S.C_theta(Q4)).*d2r));
        S.C_phi(Q5) = atan(tan(S.C_psiP(Q5).*d2r) ...
            .*sin(abs(180-S.C_theta(Q5)).*d2r));
        S.C_phi(Q6) = atan(tan((S.C_psiP(Q6)-360).*d2r) ...
            .*sin(abs(180-S.C_theta(Q6)).*d2r));
        S.C_phi(Q7) = atan(tan(S.C_psiP(Q7).*d2r) ...
            .*sin((360-S.C_theta(Q7)).*d2r));
        S.C_phi(Q8) = atan(tan((S.C_psiP(Q8)-360).*d2r) ...
            .*sin((360-S.C_theta(Q8)).*d2r));
        S.C_phi(Q9) = atan(tan(S.C_phiP(Q9).*d2r) ...
            .*cos((360-S.C_theta(Q9)).*d2r));
        % Side effect: already taken into account
        
    case 'proj_phi_only'
        % Segment
        Q1 = logical(  0 <= S.C_theta & S.C_theta <   90);
        Q2 = logical( 90 <  S.C_theta & S.C_theta <  270);
        Q3 = logical(270 <  S.C_theta & S.C_theta <= 360);  
        % Compute
        S.C_phi(Q1) = atan(tan(S.C_phiP(Q1).*d2r) ...
            .*cos(S.C_theta(Q1).*d2r));     
        S.C_phi(Q2) = atan(tan(S.C_phiP(Q2).*d2r) ...
            .*cos(abs(180-S.C_theta(Q2)).*d2r));
        S.C_phi(Q3) = atan(tan(S.C_phiP(Q3).*d2r) ...
            .*cos((360-S.C_theta(Q3)).*d2r));
        % Side effect: already taken into account
        
    otherwise
        error('TGL_PHI can only be either ''euler'' or ''proj''');
end
% Minus sign for correct rotation direction
S.C_phi = S.C_phi.*(-1);

% Convert to radians (Phi in radians already) -----------------------------
S.C_zeta = S.C_zeta.*d2r;
S.C_theta = S.C_theta.*d2r;

function S = LOCAL_Calculate3DBasePoints(S)
%% function S = LOCAL_Calculate3DBasePoints(S)

d2r = pi/180;
SIDE = logical(S.wname(:,1) == 'R'); % Computes right-side base-points
EYE = ones(size(S.wname,1),1);

% Radius of base-point
Rbp = sqrt(1./( ...
    (cos(S.C_BP_th.*d2r)).^2.*(sin(S.C_BP_phi.*d2r).^2)./(S.E_R(1)^2) + ...
    (sin(S.C_BP_th.*d2r)).^2.*(sin(S.C_BP_phi.*d2r).^2)./(S.E_R(2)^2) + ...
    (cos(S.C_BP_phi.*d2r).^2)./(S.E_R(3)^2)));

% w_ellipsoid
BP_x = Rbp.*cos(S.C_BP_th.*d2r).*sin(S.C_BP_phi.*d2r);
BP_y = Rbp.*sin(S.C_BP_th.*d2r).*sin(S.C_BP_phi.*d2r);
BP_z = Rbp.*cos(S.C_BP_phi.*d2r);

% Construct ellipsoid rotation matrix
% (see calculation in LOCAL_ConstructWhisker3D)
c_x = cos(S.E_OA(3)*d2r);
s_x = sin(S.E_OA(3)*d2r);
c_y = cos(S.E_OA(2)*d2r);
s_y = sin(S.E_OA(2)*d2r);
c_z = cos(S.E_OA(1)*d2r);
s_z = sin(S.E_OA(1)*d2r);
A = [...
    c_y*c_z, c_z*s_x*s_y - c_x*s_z, s_x*s_z + c_x*c_z*s_y; ...
    c_y*s_z, c_x*c_z + s_x*s_y*s_z, c_x*s_y*s_z - c_z*s_x; ...
    -s_y,    c_y*s_x,               c_x*c_y];

% Rotate
S.C_baseX = BP_x.*A(1,1)+BP_y.*A(1,2)+BP_z.*A(1,3);
S.C_baseY = BP_x.*A(2,1)+BP_y.*A(2,2)+BP_z.*A(2,3);
S.C_baseZ = BP_x.*A(3,1)+BP_y.*A(3,2)+BP_z.*A(3,3);

% Translate
S.C_baseX = S.C_baseX + EYE.*S.E_C(1);
S.C_baseY = S.C_baseY + EYE.*S.E_C(2);
S.C_baseZ = S.C_baseZ + EYE.*S.E_C(3);

% Left-side points are just mirror of right-side points across y-axis
S.C_baseX(~SIDE) = S.C_baseX(~SIDE).*(-1);

disp(S.C_baseX)

function [x,y,z] = LOCAL_ConstructWhisker3D(S)
%% function [x,y,z] = LOCAL_ConstructWhisker3D(S)

% Initialize whisker matricies
x = zeros(size(S.wname,1),S.Npts);
y = zeros(size(S.wname,1),S.Npts);
z = zeros(size(S.wname,1),S.Npts);

% Determine rotation matrix multiplication factors:
% (Code used to determine the matrix:)
% syms c_x s_x c_y s_y c_z s_z
% B = [1 0 0;0 c_x -s_x; 0 s_x c_x];  % Rotation about x
% C = [c_y 0 s_y; 0 1 0; -s_y 0 c_y]; % Rotation about y
% D = [c_z -s_z 0; s_z c_z 0; 0 0 1]; % Rotation about z
% A = D*C*B;
% A =
% [ c_y*c_z, c_z*s_x*s_y - c_x*s_z, s_x*s_z + c_x*c_z*s_y]
% [ c_y*s_z, c_x*c_z + s_x*s_y*s_z, c_x*s_y*s_z - c_z*s_x]
% [    -s_y,               c_y*s_x,               c_x*c_y]

% Setup useful vectors
zz = zeros(1,S.Npts);
EYE = ones(1,S.Npts);

% Determine target x and parabolic shape, 
% then rotate and translate the whisker
x_target = LOCAL_GetXfromAandS(S.C_a,S.C_s);
for ii = 1:size(S.wname,1),
    
    % Construct whisker in standard orientation:
    % (in x-y plane, negative curvature)
    xx = 0:(x_target(ii)/(S.Npts-1)):x_target(ii);
    yy = -1.*(S.C_a(ii)).*xx.^2;
    
    % Matrix multiplication
    c_x = cos(S.C_zeta(ii));
    s_x = sin(S.C_zeta(ii));
    c_y = cos(S.C_phi(ii));
    s_y = sin(S.C_phi(ii));
    c_z = cos(S.C_theta(ii));
    s_z = sin(S.C_theta(ii));
    A = [...
        c_y*c_z, c_z*s_x*s_y - c_x*s_z, s_x*s_z + c_x*c_z*s_y; ...
        c_y*s_z, c_x*c_z + s_x*s_y*s_z, c_x*s_y*s_z - c_z*s_x; ...
        -s_y,    c_y*s_x,               c_x*c_y];

    % Rotate
    x(ii,:) = xx.*A(1,1)+yy.*A(1,2)+zz.*A(1,3);
    y(ii,:) = xx.*A(2,1)+yy.*A(2,2)+zz.*A(2,3);
    z(ii,:) = xx.*A(3,1)+yy.*A(3,2)+zz.*A(3,3);
    
    % Translate
    x(ii,:) = x(ii,:) + EYE.*S.C_baseX(ii);
    y(ii,:) = y(ii,:) + EYE.*S.C_baseY(ii);
    z(ii,:) = z(ii,:) + EYE.*S.C_baseZ(ii);  
end

function x_target = LOCAL_GetXfromAandS(a,s)
%% function x_target = LOCAL_GetXfromAandS(a,s)
% Portions of this code were adapted from Blythe Towal

EYE = ones(size(a,1),1);
p00 = -1.4360;
p01 = 1.2730;
p02 = -0.013460;
p03 = 2.4970e-04;
p04 = -1.780e-06;
p05 = 4.7140e-09;
p10 = 1.3370e+02;
p11 = -7.2740;
p12 = -0.16330;
p13 = 1.1820e-04;
p14 = 6.8420e-08;
p20 = -4334;
p21 = 72.160;
p22 = 1.6440;
p23 = -7.6010e-04;
p30 = 62930;
p31 = -5.6520e+02;
p32 = -4.8130;
p40 = -407600;
p41 = 1706;
p50 = 966000;

x_target = p00.*EYE + p10.*a + p01.*s + p20.*(a.^2) + p11.*a.*s +...
    p02.*(s.^2) + p30.*(a.^3) + p21.*(a.^2).*s + p12.*a.*(s.^2) + ...
    p03.*(s.^3) + p40.*(a.^4) + p31.*(a.^3).*s + p22.*(a.^2).*(s.^2) + ...
    p13.*a.*(s.^3) + p04.*(s.^4) + p50.*(a.^5) + p41.*(a.^4).*s + ...
    p32.*(a.^3).*(s.^2) + p23.*(a.^2).*(s.^3) + p14.*a.*(s.^4) + ...
    p05.*(s.^5);
