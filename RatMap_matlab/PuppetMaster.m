function [X,Y,Z,S] = PuppetMaster(varargin)
% function [X,Y,Z,S] = PuppetMaster(varargin)
%
% This function essentially interprets high order inputs such as spread and
% absolute or relative theta angles into low order inputs and then calls
% Get_RatMap.
%
% Inputs are based on the ('param_Name',param_value) format.
% Parameters:
%
%   'colID'     - 1 x n vector identifying which columns will have
%                 theta defined (n = 1,2,3,or 7)
%                   values are anywhere from 0 (greek column) to 6,
%                   specified in ascending order
%   'theta'     - m x n array identifying the absolute theta angles (in
%                 degrees) for n number of columns (n = 1,2,3,or 7) for m time steps
%   'TGL_REL_TH'- when put to 1, the absolute theta angles defined in
%                 'theta' are treated as angles relative to the resting
%                 RatMap theta angles (default 0)
%   'spacing'   - m x p array identifying the theta angle differenc between
%                 adjacent columns (p = 1 or 6) for m time steps
%                   * linear spacing is assumed unless 6 spacing values or
%                     7 theta values are defined
%                   * when p = 6, the first value is the spacing between
%                     the 0 and 1 columns
%   'whisklength'   - how many time steps long the sweep is (default 1)
%
%       NOTE:  m (in 'theta' or 'spacing') can be either:
%                 m - the number of time steps defined
%                 1 - in which case the value is held constant over the whisk
%                 2 - defining a span, linearly interpolated over the whisk
%              If m ~= defined whisklength,1 or 2, it will either be cut short
%                 at whisklength or the final value will be repeated
%                 until it is as long as whisklength
%
%       ADVANCED NOTE:  'theta' can also be defined as a cell array with
%                       each cell element describing one whisker column.
%                       In this manner, each column can be defined with a
%                       separate m value.
%
% Parameters (continued):
%
%   'wselect'   - select the whiskers the same way in Get_RatMap
%   'Npts'      - select the number of points per whisker as in
%                 Get_RatMap
%   'alt_rest'  - flag to use the resting phi elevation values as described
%                 in Knutsen et al., 2008 as opposed to Towal et al., 2011.
%                 (default 1).  The number can be anywhere from 0 to 1 to
%                 define stages between the two.
%   'roll'      - flag to implement roll (defined by Knutsen et al 2008) (default 1)
%   'elev'      - flag to implement elevation (defined by Knutsen et al 2008) (default 1)
%   'plot'      - flag to plot the resulting configuration (default 0)
%   'head'      - flag to plot the rat head in the plot (default 0)
%   'save'      - turns on the flag to save the video, parameter value is the
%                 name of the video to be saved
%   'eparams'   - add any extra parameters found in Get_RatMap but not
%                 PuppetMaster, 'eparams' parameter is cell that bundles
%                 the new parameters in the same format (example: {'W_a',0}
%                 for straight whiskers)
%
% Outputs:
%       X = n x m x l matrix giving the X-coordinate for all m nodes of
%           every n whiskers for all l time steps
%       Y = n x m x l matrix giving the Y-coordinate for all m nodes of
%           every n whiskers for all l time steps
%       Z = n x m x l matrix giving the z-coordinate for all m nodes of
%           every n whiskers for all l time steps
%       S = a struct array of size l giving RatMap data for every time step
%
%
% Example Function Calls:
% % hold the greek column at rest and protract the front whiskers 30 degrees
% [X,Y,Z,S] = PuppetMaster('colID',[0 6],'theta',[0 30],'TGL_REL_TH',1);
%
% % input tracked data for the greek and fourth columns
% [X,Y,Z,S] = PuppetMaster('colID',[0 4],'theta',[back_whisker_data(1:500)' front_whisker_data(1:500)')
%
% % hold the second column at rest and vary the spacing from 0 to 50/6 over 10 time steps
% [X,Y,Z,S] = PuppetMaster('colID',2,'theta',0,'TGL_REL_TH',1,'spacing',[0 50/6],'whisklength',10);
% 
%
% Lucie Huet
% Oct.  5, 2012


%% Handle Inputs

% Defaults:
colID = [];
Theta = [];
TGL_REL_TH = 0;
SPREAD = [];
whisklength = 1;
wselect = '';
KnutsenFlag = 1;
rollflag = 1;
elevflag = 1;
plotflag = 0;
Npts = 100;
plothead = 0;
savevidflag = 0;
vidname = '';
eparams = {};


% Accept Inputs
settings = varargin(1:end);
for ii = 1:2:length(settings)
    switch settings{ii}
        case 'colID',       colID = settings{ii+1}; % which columns you are identifying (default [])
            % check if well defined
            if sum(colID<0 | colID>6)>0
                error('colID poorly defined: one or more columns outside of 0:6')
            end
            % check if defined in order
            if sum(diff(colID,1,2)<0)>0
                error('colID must be defined in order from back to front')
            end
        case 'theta',       Theta = settings{ii+1};
        case 'TGL_REL_TH',  TGL_REL_TH = settings{ii+1};
        case 'spacing',     SPREAD = settings{ii+1};
        case 'whisklength', whisklength = settings{ii+1};
        case 'wselect',     wselect = settings{ii+1};
        case 'Npts',        Npts = settings{ii+1};
        case 'alt_rest',    KnutsenFlag = settings{ii+1};
        case 'roll',        rollflag = settings{ii+1};
        case 'elev',        elevflag = settings{ii+1};
        case 'plot',        plotflag = settings{ii+1};
        case 'head',        plothead = settings{ii+1};
        case 'save',        savevidflag = 1; plotflag = 1; vidname = settings{ii+1};
        case 'eparams',     eparams = settings{ii+1};
        otherwise
            warning('PuppetMaster:FaultyInput',[settings{ii},' is not a valid setting - ignoring input']);
    end
end

% check eparams for Npts
for ii = 1:2:length(eparams)
    switch eparams{ii}
        case 'Npts',        Npts = eparams{ii+1};
    end
end

%% Manipulate inputs until they are well defined

% handle case where Theta is cell array (input single and array inputs to
% different columns)
if iscell(Theta)
    T_len = length(Theta);
    L = 1;
    for ii = 1:T_len
        if length(Theta{ii})>L, L = length(Theta{ii}); end
    end
    THETA = zeros(L,T_len);
    for ii = 1:T_len
        if size(Theta{ii},2)~=1, Theta{ii} = Theta{ii}'; end %#ok<AGROW> % be sure element is column vector
        switch size(Theta{ii},1)
            case L,     THETA(:,ii) = Theta{ii};
            case 2,     THETA(:,ii) = linspace(Theta{ii}(1),Theta{ii}(2),L);
            case 1,     THETA(:,ii) = repmat(Theta{ii},L,1);
            otherwise
                warning('PuppetMaster:FaultyInput','One theta column poorly defined - extending final value')
                THETA(:,ii) = [Theta{ii}; repmat(Theta{ii},whisklength - size(Theta{ii},2),1)];
        end
    end
else
    THETA = Theta;
end

% check got both or neither colID and theta input
if sum([isempty(colID) isempty(THETA)])==1
    warning('PuppetMaster:FaultyInput','Both colID and Theta need to be defined - ignoring input')
    colID = [];
    THETA = [];
end

% check colID has good number of columns and colID and theta match
if ~isempty(colID) % only enter loop if theta input exists
    if size(colID,2)~=1 && size(colID,2)~=2 && size(colID,2)~=3 && size(colID,2)~=7
        if size(colID,1)==1 || size(colID,1)==2 || size(colID,1)==3 || size(colID,1)==7
            colID = colID';
        else 
            error('colID needs to have length == 1,2,3,or 7');
        end
    end
    
    if size(colID,2)~=size(THETA,2)
        if size(colID,2)==size(THETA,1)
            THETA = THETA';
        else
            error('Number of thetas defined does not match number of columns defined')
        end
    end
end

% check SPREAD has good number of columns
if size(SPREAD,2)~=0 && size(SPREAD,2)~=1 && size(SPREAD,2)~=6
    if size(SPREAD,1)==0 || size(SPREAD,1)==1 || size(SPREAD,1)==6
        SPREAD = SPREAD';
    else
        error('Incorrect number of spreads defined')
    end
end

% determine whisklength
if whisklength == 1
    whisklength = max([1 size(SPREAD,1) size(THETA,1) size(colID,1)]);
end

% make all arrays same length
if size(colID,1) ~= 1 && size(colID,1) ~= 0
    error('colID must be a vector and not an array')
end

switch size(SPREAD,1)
    case {0 whisklength}, % do nothing
    case 1
        SPREAD = repmat(SPREAD,whisklength,1);
    case 2
        spread = SPREAD;
        SPREAD = zeros(whisklength,size(spread,2));
        for ii = 1:size(SPREAD,2)
            SPREAD(:,ii) = linspace(spread(1,ii),spread(2,ii),whisklength);
        end
    otherwise
        if size(SPREAD,1)>whisklength
            warning('PuppetMaster:FaultyInput','Spacing defined too long - trunkating for whisklength')
            SPREAD = SPREAD(1:whisklength,:);
        else
            warning('PuppetMaster:FaultyInput','Spacing defined too short - extending final theta value')
            SPREAD = [SPREAD; repmat(SPREAD(end,:),whisklength - size(SPREAD,2),1)];
        end
end

switch size(THETA,1)
    case {0 whisklength}, % do nothing
    case 1
        THETA = repmat(THETA,whisklength,1);
    case 2
        theta = THETA;
        THETA = zeros(whisklength,size(theta,2));
        for ii = 1:size(THETA,2)
            THETA(:,ii) = linspace(theta(1,ii),theta(2,ii),whisklength);
        end
    otherwise
        if size(THETA,1)>whisklength
            warning('PuppetMaster:FaultyInput','Theta defined too long - truncating for whisklength')
            THETA = THETA(1:whisklength,:);
        else
            warning('PuppetMaster:FaultyInput','Theta defined too short - extending final theta value')
            THETA = [THETA; repmat(THETA(end,:),whisklength - size(THETA,1),1)];
        end
end

%% Prepare original whisker values

% get original RatMap values
[~,~,~,S0] = Get_RatMap(wselect);
Orig_th = S0.C_thetaP';
Orig_z = S0.C_zetaP';
Orig_phi = S0.C_phi'.*(-180/pi);

NOM_TH_EQ = S0.EQ_W_th; % [(theta diff between columns) 0 (absolute theta of column 'behind' greek column)]
NOM_TH_EQ = repmat(NOM_TH_EQ,whisklength,1);
NEW_TH_EQ = NOM_TH_EQ;

NOM_TH = repmat(NOM_TH_EQ(:,3),1,7) + NOM_TH_EQ(:,1)*(1:7);
NEW_TH = zeros(whisklength,7);

% translate relative angles into absolute ones
if TGL_REL_TH
    THETA = THETA + NOM_TH(:,colID+1);
end

%% Calculate new thetas

case_handle = num2str([size(THETA,2) size(SPREAD,2)]);

switch case_handle
    case '0  0' % no spacing or theta defined
        NEW_TH = NOM_TH;
    case '0  1' % only one spacing defined - keep gamma column constant
        NEW_TH_EQ(:,1) = SPREAD;
        NEW_TH = repmat(NOM_TH(:,1),1,7) + NEW_TH_EQ(:,1)*(0:6);
    case '0  6' % all spacings defined, no theta - keep gamma column constant
        S = [zeros(size(SPREAD,1),1) SPREAD];
        for ii = 1:7
            NEW_TH(:,ii) = NOM_TH(:,1) + sum(S(:,1:ii),2);
        end
    case '1  0' % only one theta defined
        NEW_TH_EQ(:,3) = THETA - NOM_TH_EQ(:,1)*(colID+1);
        NEW_TH = repmat(NEW_TH_EQ(:,3),1,7) + NEW_TH_EQ(:,1)*(1:7);
    case '1  1' % one theta and one spread defined
        NEW_TH_EQ(:,1) = SPREAD;
        NEW_TH_EQ(:,3) = THETA - NEW_TH_EQ(:,1)*(colID+1);
        NEW_TH = repmat(NEW_TH_EQ(:,3),1,7) + NEW_TH_EQ(:,1)*(1:7);
    case '1  6' % one theta and all spreads defined
        t = (1:7)-(colID+1);
        S = [SPREAD(:,1:colID) zeros(size(SPREAD,1),1) SPREAD(:,colID+1:end)];
        for ii = 1:7
            NEW_TH(:,ii) = THETA + sign(t(ii))*sum(S(:,(colID+1):sign(t(ii)):(colID+1+t(ii))),2);
        end
    case '2  0' % only two thetas defined
        NEW_TH_EQ(:,1) = diff(THETA,1,2)./diff(colID);
        NEW_TH_EQ(:,3) = THETA(:,1)-NEW_TH_EQ(:,1)*(colID(1)+1);
        NEW_TH = repmat(NEW_TH_EQ(:,3),1,7) + NEW_TH_EQ(:,1)*(1:7);
    case '2  1' % only two thetas and one spread defined
        warning('PuppetMaster:OverdefinedInput','Two thetas defined - using linear spacing')
        NEW_TH_EQ(:,1) = diff(THETA,1,2)./diff(colID);
        NEW_TH_EQ(:,3) = THETA(:,1)-NEW_TH_EQ(:,1)*(colID(1)+1);
        NEW_TH = repmat(NEW_TH_EQ(:,3),1,7) + NEW_TH_EQ(:,1)*(1:7);
    case '2  6' % two thetas and all spreads defined
        warning('PuppetMaster:OverdefinedInput','Two thetas defined - scaling spacing input')
        % resize spread
        a = diff(THETA,1,2)./sum(SPREAD(:,colID(1)+1:colID(2)),2);
        SPREAD = SPREAD.*repmat(a,1,6);
        % perform as before
        t = (1:7)-(colID(1)+1);
        S = [SPREAD(:,1:colID(1)) zeros(size(SPREAD,1),1) SPREAD(:,colID(1)+1:end)];
        for ii = 1:7
            NEW_TH(:,ii) = THETA(:,1) + sign(t(ii))*sum(S(:,(colID(1)+1):sign(t(ii)):(colID(1)+1+t(ii))),2);
        end
    case '3  0' % three thetas defined
        D1 = diff(THETA(:,1:2),1,2)./(colID(2) - colID(1));
        backth = THETA(:,1)- colID(1)*D1;
        NEW_TH(:,1:(colID(2)+1)) = repmat(backth,1,colID(2)+1) + D1*(0:colID(2));
        D2 = diff(THETA(:,2:3),1,2)./(colID(3) - colID(2));
        NEW_TH(:,(colID(2)+1):end) = repmat(THETA(:,2),1,(7 - colID(2))) + D2*(0:(6 - colID(2)));
    case '3  1' % three thetas and one spread defined
        warning('PuppetMaster:PoorlyDefinedInput','Three thetas defined - using linear spacing')
        D1 = diff(THETA(:,1:2),1,2)./(colID(2) - colID(1));
        backth = THETA(:,1)- colID(1)*D1;
        NEW_TH(:,1:(colID(2)+1)) = repmat(backth,1,colID(2)+1) + D1*(0:colID(2));
        D2 = diff(THETA(:,2:3),1,2)./(colID(3) - colID(2));
        NEW_TH(:,(colID(2)+1):end) = repmat(THETA(:,2),1,(7 - colID(2))) + D2*(0:(6 - colID(2)));
    case '3  6' % three thetas and one spread defined
        warning('PuppetMaster:OverdefinedInput','Three thetas defined - scaling spacing input')
        % resize both spread halves
        a = diff(THETA(:,1:2),1,2)./sum(SPREAD(:,colID(1)+1:colID(2)),2);
        SPREAD(:,1:colID(2)) = SPREAD(:,1:colID(2)).*repmat(a,1,colID(2));
        b = diff(THETA(:,2:3),1,2)./sum(SPREAD(:,colID(2)+1:colID(3)),2);
        SPREAD(:,(colID(2)+1):end) = SPREAD(:,(colID(2)+1):end).*repmat(b,1,(6-colID(2)));
        % perform as before
        t = (1:7)-(colID(1)+1);
        S = [SPREAD(:,1:colID(1)) zeros(size(SPREAD,1),1) SPREAD(:,colID(1)+1:end)];
        for ii = 1:7
            NEW_TH(:,ii) = THETA(:,1) + sign(t(ii))*sum(S(:,(colID(1)+1):sign(t(ii)):(colID(1)+1+t(ii))),2);
        end
    case '7  0' % all theta defined
        NEW_TH = THETA(:,colID+1);
    case '7  1' % all theta defined
        warning('PuppetMaster:OverdefinedInput','All thetas already defined - ignoring spacing input')
        NEW_TH = THETA(:,colID+1);
    case '7  6' % all theta defined
        warning('PuppetMaster:OverdefinedInput','All thetas already defined - ignoring spacing input')
        NEW_TH = THETA(:,colID+1);
    otherwise
        error('Bad Input - incorrect number of thetas and spreads defined')
end

%% Calculate phi and zeta angles

% KNUTSEN RESTING PHI - DO NOT CHANGE VALUES
if KnutsenFlag
    Knutsen_phi = zeros(size(Orig_phi));
    for ii = 1:size(S0.wname,1)
        switch S0.wname(ii,2)
            case 'A', Knutsen_phi(ii) =  56;
            case 'B', Knutsen_phi(ii) =  25;
            case 'C', Knutsen_phi(ii) = - 4.2;
            case 'D', Knutsen_phi(ii) = -27.2;
            case 'E', Knutsen_phi(ii) = -44;
        end
    end
    Orig_phi = Orig_phi + KnutsenFlag*(Knutsen_phi - Orig_phi);
end

% get NEW_TH_ALL in whisker order
NEW_TH_ALL = zeros(whisklength,size(S0.wname,1));
for ii = 1:size(S0.wname,1)
    switch S0.wname(ii,3)
        case '0', NEW_TH_ALL(:,ii) = NEW_TH(:,1);
        case '1', NEW_TH_ALL(:,ii) = NEW_TH(:,2);
        case '2', NEW_TH_ALL(:,ii) = NEW_TH(:,3);
        case '3', NEW_TH_ALL(:,ii) = NEW_TH(:,4);
        case '4', NEW_TH_ALL(:,ii) = NEW_TH(:,5);
        case '5', NEW_TH_ALL(:,ii) = NEW_TH(:,6);
        case '6', NEW_TH_ALL(:,ii) = NEW_TH(:,7);
    end
end

dth = NEW_TH_ALL - repmat(Orig_th,whisklength,1);

% get new phi and zetas in whisker order - from Knutsen et al 2008 - DO NOT CHANGE VALUES
ZT = zeros(2,size(S0.wname,1));
PT = zeros(1,size(S0.wname,1));
for ii = 1:size(S0.wname,1)
    switch S0.wname(ii,2)
        case 'A', ZT(:,ii) = [ 0;      -0.76];       PT(ii) =  0.12;
        case 'B', ZT(:,ii) = [ 0;      -0.25];       PT(ii) =  0.30;
        case 'C', ZT(:,ii) = [ 0;       0.22];       PT(ii) =  0.30;
        case 'D', ZT(:,ii) = [ 0;       0.42];       PT(ii) =  0.14;
        case 'E', ZT(:,ii) = [ 0;       0.73];       PT(ii) = -0.02;
    end
end

New_phi = repmat(Orig_phi,whisklength,1) + dth.*repmat(PT,whisklength,1)*elevflag;
New_z = repmat(Orig_z,whisklength,1) + (dth.*repmat(ZT(2,:),whisklength,1) + ...
    (dth.^2).*repmat(ZT(1,:),whisklength,1))*rollflag;

X = zeros(size(S0.wname,1),Npts,whisklength);
Y = X;
Z = X;

%% Generate output and plot

clear S

for ii = 1:whisklength
    % generate RatMap
    [X(:,:,ii),Y(:,:,ii),Z(:,:,ii),S(ii)] = Get_RatMap(wselect,...
        'W_th',NEW_TH_ALL(ii,:)','W_zeta',New_z(ii,:),...
        'TGL_PHI','euler','W_phiE',New_phi(ii,:),...
        'Npts',Npts,eparams{:}); %#ok<AGROW>
    
    % plot results
    if plotflag
        clf;
        hold on
        
        % plot each whisker
        for jj = 1:size(X,1)
            plot3(X(jj,:,ii),Y(jj,:,ii),Z(jj,:,ii))
        end
        
        % plot head if asked for
        if plothead
            Plot_RatMap_Head;
            hold off
        end
        
        axis equal
        if length(wselect)>=1 && strcmp(wselect(1),'L')
            xlim([-60 0]);
        elseif length(wselect)>=1 && strcmp(wselect(1),'R')
            xlim([0 60]);
        else
            xlim([-60 60]);
        end
        
        ylim([-50 20]); zlim([-50 15]);
        view(215,25) %(90,0) %(-180,90)
        
        drawnow
        
        if savevidflag %#ok<*MSNU>
            mov(ii) = getframe(gcf);  %#ok<AGROW>
        end
    end
end

if savevidflag && plotflag
    movie2avi(mov,vidname);
end