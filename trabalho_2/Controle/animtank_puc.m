function [sys,x0]=animtank_puc(t,x,u,flag,ts) %#ok<INUSL>
%ANIMTANK Animation of water tank system.
%

% Copyright 1994-2014 The MathWorks, Inc.

global tankdemo

if flag==2,
    if any(get(0,'Children')==tankdemo),
        if strcmp(get(tankdemo,'Name'),'Tank Demo'),
            
            % Update tank one level
            tankHndlList=get(tankdemo,'UserData');
            yData=get(tankHndlList(1),'YData');
            yOffset=yData(1);
            yData(3:4)=[1 1]*u(2)+yOffset;
            set(tankHndlList(1),'YData',yData);
            
            yData=get(tankHndlList(2),'YData');
            yData([3 4])=[1 1]*u(2)+yOffset;
            set(tankHndlList(2),'YData',yData);
            
            yData=[1 1]*u(1)+1;
            set(tankHndlList(3),'YData',yData);
            
            % Update tank 2 one level
            tankHndlList=get(tankdemo,'UserData');
            yData=get(tankHndlList(4),'YData');
            yOffset=yData(1);
            yData(3:4)=[1 1]*u(3)+yOffset;
            set(tankHndlList(4),'YData',yData);
            
            yData=get(tankHndlList(5),'YData');
            yData([3 4])=[1 1]*u(3)+yOffset;
            set(tankHndlList(5),'YData',yData);
            
            drawnow;
        end
    end
    sys=[];
    x0=[];
    
elseif flag == 4 % Return next sample hit
    
    % ns stores the number of samples
    ns = t/ts;
    
    % This is the time of the next sample hit.
    sys = (1 + floor(ns + 1e-13*(1+ns)))*ts;
    x0=[];
    
elseif flag==0,
    
    % Initialize the figure for use with this simulation
    fuzzy_animinit('Tank Demo');
    tankdemo=findobj(0,'Name','Tank Demo');
    %set_param('sltankrule_puc','MaxConsecutiveZCsMsg','none');
    tank1Wid=0.5;
    tank1Ht=2;
    tank1Init=0;
    setPt=5;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tankX=[0 0 1 1]-0.5;
    tankY=[1 0 0 1];
    % Draw the tank
    line(1.05*tankX*tank1Wid+1,tankY*tank1Ht+0.95,'LineWidth',2,'Color','black');
    tankX=[0 1 1 0 0]-0.5;
    tankY=[0 0 1 1 0];
    % Draw the water
    waterX=tankX*tank1Wid+1;
    waterY=tankY*tank1Init + 0.95;
    tank1Hndl=patch(waterX,waterY,'blue','EdgeColor','none');
    % Draw the gray wall
    waterY([1 2 5])=tank1Ht*[1 1 1]+1;
    waterY([3 4])=tank1Init*[1 1]+1;
    tank2Hndl=patch(waterX,waterY,[.9 .9 .9],'EdgeColor','none');
    % Draw the set point
    %lineHndl=line([0 0.5],setPt*[1 1]+1,'Color','red','LineWidth',4);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
     d=1.2;
    tankX=[0 0 1 1]-0.5;
    tankY=[1 0 0 1];
    % Draw the tank
    line(1.05*tankX*tank1Wid+1 + d,tankY*tank1Ht+0.95,'LineWidth',2,'Color','black');
    tankX=[0 1 1 0 0]-0.5;
    tankY=[0 0 1 1 0];
    % Draw the water
    waterX=tankX*tank1Wid+1;
    waterY=tankY*tank1Init+0.95;
    tank21Hndl=patch(waterX + d, waterY,'blue','EdgeColor','none');
    % Draw the gray wall
    waterY([1 2 5])=tank1Ht*[1 1 1]+1;
    waterY([3 4])=tank1Init*[1 1]+1;
    tank22Hndl=patch(waterX + d, waterY,[.9 .9 .9],'EdgeColor','none');
    % Draw the set point
    lineHndl=line([0 0.1] + 2*d, setPt*[1 1]+1,'Color','red','LineWidth',4);
    
    h=0.1;
    tankX=[0 0 1 1 0]-0.5;
    tankY=[0 1 1 0 0];
    line(tankX*(d-tank1Wid)+1 + d/2,tankY*h+0.95,'LineWidth',2,'Color','black');
    waterX=tankX*tank1Wid*1.5 + 0.4 ;
    waterY=tankY*h+0.95;
    connectHndl=patch(waterX+ d,waterY,'blue','EdgeColor','none');
        
    
    line([-1 0.75],[1.3 1.3],'LineWidth',2,'Color','black');
    cx=0.325;
    cy=1.3;
    m=0.2;
    line([cx-m cx-m cx+m cx+m cx-m],[cy+m cy-m cy+m cy-m cy+m],'LineWidth',2,'Color','black');
    line([2.45 2.8 2.8],[1 1 0.5],'LineWidth',8,'Color','black');
    line([2 2.8 2.8],[1 1 0.5],'LineWidth',5,'Color','blue');
    
    line([2.3 2.8 2.8]-0.7,[1 1 0.7]+2.4,'LineWidth',8,'Color','black');
    line([2.3 2.8 2.8]-0.7,[1 1 -1.4]+2.4,'LineWidth',5,'Color','blue');
    
    set(gcf, ...
        'Color',[.9 .9 .9], ...
        'UserData',[tank1Hndl tank2Hndl lineHndl tank21Hndl tank22Hndl]);
    set(gca, ...
        'XLim',[0 3],'YLim',[0 4], ...
        'XColor','black','YColor','black', ...
        'Box','on');
    axis equal
    xlabel('Tank 1           Tank 2','Color','black','FontSize',14);
    set(get(gca,'XLabel'),'Visible','on')
    
    sys=[0 0 0 3 0 0];
    x0=[];
    
end;
