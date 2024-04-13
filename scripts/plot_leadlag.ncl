;**************************************
;var_cam.ncl
;**************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"
load "/glade/work/aherring/CESM2/arh-git-scripts/ncl/se_func_lib.ncl"
;**************************************
begin
;****************************************
  p0 = 1000.d2
  secpday = 60.*60.*24;
  pi = 3.14159265358979
  g = 9.80616
  Re = 6371.22
  cp = 1004.
  Rd = 287.
;****************************************
  onemnth = False

  plotTitle = "ARCTICGRIS_map_TO_f19"

  ;;;Model output
  dir = "/glade/work/awaling/AR_GrIS/TempestRemap/f19/land/"
  dir_scrip = "../data/scrip/lnddata/"
  fname = (/"cam6_2_022.se_FHIST_ne0np4.ARCTICGRIS.ne30x8_mt12_7680pes_200510_mg3-Nx1yrs"/)
  nfiles = dimsizes(fname)

  ;;;GrIS blob filter
  datadir = "/glade/work/awaling/AR_GrIS/TempestRemap/f19/f19_GrIS_blobs/cam6_2_022.se_FHIST_ne0np4.ARCTICGRIS.ne30x8_mt12_7680pes_200510_mg3-Nx1yrs/"
  datadir_scrip = "../data/scrip/"
  tfile = (/"ARCTICGRIS_map_TO_f19_tcenters.nc"/)
  cfile = (/"ARCTICGRIS_map_TO_f19_cmasks.nc"/)

  rdir = "/"
  fcomp = "clm2"
  fincl = "h1"

  VAR = "RAIN_ICE"
  factor = secpday
  lbtitle = "RAIN+SNOW (mm/day)"
  maxcn  =   20.
  mincn  =    0.  

  cnlevs = 20.
  cnspc  = (maxcn-mincn)/cnlevs

  anomaly = False

  cellfill = True

;******************************************************

  ;fdate = "."+fcomp+"."+fincl+".*"

  if (fcomp.eq."cam") then
    nframes = 9
    runnams = (/"t-24hr","t-18hr","t-12hr","t-6hr","t","t+6hr","t+12hr","t+18hr","t+24hr"/)
  else
    nframes = 7
    runnams = (/"t-3days","t-2days","t-1days","t","t+1days","t+2days","t+3days"/)
  end if

  pltNam = "temp_precipex"

;******************************************************
; create plot
;******************************************************
  wkstype = "pdf"
  wks = gsn_open_wks(wkstype,pltNam)               ; open workstation
  ;;gsn_define_colormap(wks,"cmp_b2r")
  gsn_define_colormap(wks,"cmocean_dense")
  ;gsn_reverse_colormap(wks)
  plot = new( (/nfiles*nframes/), "graphic")

   res             = True
   res@gsnMaximize = False
   res@gsnDraw     = False      ; Will draw later, after overlaying
   res@gsnFrame    = False      ; all plots

   res@cnFillOn          = True               ; turn on color fill
   res@cnLinesOn         = False
   res@cnLineLabelsOn    = False

   res@gsnSpreadColors   = True               ; use full range of color map
   res@cnInfoLabelOn        = False
   res@lbLabelBarOn         = False
   res@tiMainFontHeightF = 0.006

   res@vpWidthF  = 0.3
   res@vpHeightF = 0.8

   res@mpGridAndLimbOn   = False           ; turn on lat/lon lines
   res@mpGridLatSpacingF = 10.
   res@mpGridLonSpacingF = 30.
   ;res@mpGridLineColor      = -1
   res@mpPerimOn         = True               ; turn off perimeter
   res@mpFillOn         = True
   ;;res@mpOutlineBoundarySets     = "NoBoundaries"

   res@mpProjection = "LambertConformal";;"Orthographic";;"Mollweide"
   res@mpLambertParallel1F = 85
   res@mpLambertParallel2F = 58
   res@mpLambertMeridianF  = -45
   res@mpLimitMode       = "LatLon"
   res@mpMinLonF = -56.4
   res@mpMaxLonF = -30.
   res@mpMinLatF = 59.4;;58.
   res@mpMaxLatF = 85;;85.

;******************************************************
; begin loop
;******************************************************

 do n = 0,nfiles-1

   files = systemfunc ("ls " + dir + fname(n) + rdir + "*.nc")
   files_scrip = systemfunc ("ls " + dir_scrip + "*.nc")
   nhist = dimsizes(files_scrip)

   f  = addfile (files_scrip(0),"r")
   dims = getfiledimsizes(f)
   dimnames = getfiledimnames(f)
   delete(f)

   if (any(dimnames.eq."lon")) then
     latlon = True
   else
     latlon = False
   end if
   delete([/dims,dimnames/])

   nstart = 0 
   if (onemnth) then
    nfinish = nstart
   else
    nfinish = nhist-1
   end if

   print("load:"+files(nstart:nfinish))
   f  = addfiles (files(nstart:nfinish),"r")
   ListSetType(f,"cat")
   delete(files)
   
   ;print(f[0])
   lat = f[0]->lat
   lon = f[0]->lon
   time = f[:]->time
   ntime = dimsizes(time)

   if (VAR.eq."QSNOMELT_ICE") then
     var = f[:]->$VAR$
     tmp = f[:]->QICE_MELT
     var = var + tmp
     delete(tmp)
   else if (VAR.eq."RAIN_ICE") then
     var = f[:]->$VAR$
     ;tmp = f[:]->SNOW_ICE
     ;var = var + tmp
     ;delete(tmp)
   else
     var = f[:]->$VAR$
   end if
   end if
   var = var*factor
   delete(f)

   ;print(tfile(n))
   sf = addfile(datadir_scrip+tfile(n),"r")
   tcent = sf->time_center
   tinds = sf->tind_center
   delete(sf)

   sf = addfile(datadir_scrip+cfile(n),"r")
   cmasks = sf->cmasks
   delete(sf)

   nevents = dimsizes(tcent)
   dims = dimsizes(var)

   if (latlon) then
     vcent = new( (/nevents,nframes,dims(1),dims(2)/), "float")
   else
     vcent = new( (/nevents,nframes,dims(1)/), "float")
   end if

   vcent@_FillValue = default_fillvalue("float")
   vcent = vcent@_FillValue

   dt = (nframes-1)/2
   do m = 0, nevents-1
     do q = 0,nframes-1

       if (fcomp.eq."cam") then
         tindnow = tinds(m)+q-dt
       else
         tindnow = ind(time.eq.round(tcent(m),0))+q-dt
       end if

       if (tindnow.lt.0 .or. tindnow.gt.(ntime-1)) then
         continue
       end if

       if (latlon) then
         vcent(m,q,:,:) = var(tindnow,:,:)*tofloat(cmasks(m,:,:))
       else
         vcent(m,q,:) = var(tindnow,:)*tofloat(cmasks(m,:))
       end if

     end do
   end do

   print("max vcent " + max(vcent))
   print("min vcent " + min(vcent))

   plotTitle = plotTitle + " (nevents="+ nevents +")"

   res@cnLevelSelectionMode = "ManualLevels"
   res@cnMinLevelValF  = mincn
   res@cnMaxLevelValF  = maxcn
   res@cnLevelSpacingF = cnspc

   if (.not.latlon) then
     res@sfXArray = lon
     res@sfYArray = lat
   end if
 
   if (cellfill) then
     if (latlon) then
       ;;;Grid descriptor
       griddir = (/"/glade/work/aherring/grids/uniform-res/f19/grids/"/)
       griddir_scrip = "../data/scrip/"
       scrip = (/"fv1.9x2.5_141008.nc"/)
     else
       griddir = (/"/glade/work/aherring/grids/uniform-res/ne30np4.pg2/grids/"/)
       griddir_scrip = "../data/scrip/"
       scrip = (/"ne30pg2_scrip_170608.nc"/)
     end if

     res@cnFillMode = "CellFill"
     vf = addfile(griddir_scrip(n)+scrip(n),"r")
     res@sfXCellBounds = vf->grid_corner_lon
     res@sfYCellBounds = vf->grid_corner_lat
     delete(vf)
   else
     res@cnFillMode = "AreaFill"
   end if 

   res@gsnCenterStringFontHeightF = 0.01

   do q = 0,nframes-1
     if (anomaly .and. q.ne.dt) then
       if (latlon) then
         vcent(:,q,:,:) = vcent(:,q,:,:) - vcent(:,dt,:,:)
       else
         vcent(:,q,:) = vcent(:,q,:) - vcent(:,dt,:)
       end if 
     end if
 
     if (latlon) then 
       bar = dim_avg_n(vcent(:,q,:,:),0)
       bar!0 = "lat"
       bar&lat = lat
       bar&lat@units = "degrees_north"
       bar!1 = "lon"
       bar&lon = lon
       bar&lon@units = "degrees_east"
     else
       bar = dim_avg_n(vcent(:,q,:),0)
     end if

     if (anomaly .and. q.eq.dt) then
       bar = 0.
     end if
     
     ;print(q)
     ;printVarSummary(bar)
     plot(q) = gsn_csm_contour_map(wks,bar,res)  
     ;plot!0 = "row" 
     ;plot!1 = "col" 
     
     ;plot1d = ndtooned(plot(col | :, row | :))
     delete(bar)
   end do

   if (.not.latlon) then
     delete([/res@sfXArray,res@sfYArray/])
   end if

   if (cellfill) then
     delete([/res@sfXCellBounds,res@sfYCellBounds/])
   end if

 end do;;n


;;;drawNDCGrid(wks)
;*****************************************************
; Plot panel
;*****************************************************

  resP                  = True                   ; modify the panel plot
  resP@gsnFrame         = False                  ; don't advance panel plot
  ;resP@gsnPaperOrientation = "portrait"

  resP@gsnPanelFigureStrings = runnams
  resP@gsnPanelFigureStringsFontHeightF = 0.012
  resP@gsnPanelFigureStringsPerimOn     = True
  resP@gsnPanelFigureStringsBackgroundFillColor = "white"
  resP@amJust           = "TopLeft"

  resP@txString = plotTitle

  resP@gsnPanelLabelBar = True
  resP@lbBoxSeparatorLinesOn= False
  resP@gsnPanelBottom   = 0.1
  resP@lbTitleString              = lbtitle
  resP@lbTitlePosition            = "Bottom"
  resP@lbTitleFontHeightF         = 0.015
  resP@lbLabelFontHeightF         = 0.012
  resP@pmLabelBarOrthogonalPosF   = -0.02
  ;print(nframes)
  ;resP@txString = pltTitle
  ;printVarSummary(wks)
  ;printVarSummary(plot)
  gsn_panel(wks,plot,(/nfiles,nframes/),resP)

frame(wks)

end

