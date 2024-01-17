;**************************************
;var_cam.ncl
;**************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"
load "/gpfs/fs1/work/aherring/CESM2/arh-git-scripts/ncl/se_func_lib.ncl"
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

  kframe = 4
  lbtitle = "PRECT (mm/day) 99th ptile, LAG = +0 hrs"
  lbtitle = "Precipitation Rate (mm/day)"

  writencdf = False
  loadncdf = True

  subdir_scrip = "../data/precip/"

  fname0 = (/"geolines_PRECT_amean_95ptile_f19.nc", \
            "geolines_PRECT_amean_95ptile_f09.nc", \
            "geolines_PRECT_amean_95ptile_ne30pg2.nc", \
            "geolines_PRECT_amean_95ptile_ne30pg3.nc", \
            "geolines_PRECT_amean_95ptile_ARCTIC.nc", \
            "geolines_PRECT_amean_95ptile_ARCTICGRIS.nc", \
            "geolines_PRECT_amean_95ptile_ERA5.nc", \
            "geolines_PRECT_amean_95ptile_MERRA2.nc"/)
  fname0 = (/"geolines_PRECT_amean_99ptile_pm1day_f19.nc", \
	     "geolines_PRECT_amean_99ptile_pm1day_f09.nc", \
             "geolines_PRECT_amean_99ptile_pm1day_ne30pg2.nc", \
             "geolines_PRECT_amean_99ptile_pm1day_ne30pg3.nc", \
             "geolines_PRECT_amean_99ptile_pm1day_ARCTIC.nc", \
             "geolines_PRECT_amean_99ptile_pm1day_ARCTICGRIS.nc", \
             "geolines_PRECT_amean_99ptile_pm1day_ERA5.nc", \
             "geolines_PRECT_amean_99ptile_pm1day_MERRA2.nc"/)


  dir = "/glade/campaign/cgd/amp/aherring/archive/"
  rdir = "/atm/hist/"
  fname = (/"cam6_2_022.se_FHIST_f19_f19_mg17_900pes_200506_mg3-Nx5yrs", \
	    "cam6_2_022.se_FHIST_f09_f09_mg17_1800pes_200507_mg3-Nx5yrs", \
	    "cam6_2_022.se_FHIST_ne30pg2_ne30pg2_mg17_1800pes_200507_mg3-Nx5yrs", \
	    "cam6_2_022.se_FHIST_ne30pg3_ne30pg3_mg17_1800pes_200507_mg3-Nx5yrs", \
	    "cam6_2_022.se_FHIST_ne0np4.ARCTIC.ne30x4_mt12_7680pes_200507_mg3-Nx2yrs", \
	    "cam6_2_022.se_FHIST_ne0np4.ARCTICGRIS.ne30x8_mt12_7680pes_200510_mg3-Nx1yrs"/)
;;	    "ERA5", \
;;	    "MERRA2"/)
  fdate = ".cam.h3.*"

  nera5 = 6
  era5dir = "/glade/campaign/cgd/amp/aherring/annelise_backup/TempestExtremes/ERA5_raw/"
  nmerra2 = 7
  merra2dir = "/glade/campaign/cgd/amp/aherring/annelise_backup/TempestExtremes/merra2_precip/"

  sdir = (/"/glade/campaign/cgd/amp/aherring/annelise_backup/TempestExtremes/ESMF/SB/f19/", \
           "/glade/campaign/cgd/amp/aherring/annelise_backup/TempestExtremes/ESMF/SB/ne30pg2/", \
           "/glade/campaign/cgd/amp/aherring/annelise_backup/TempestExtremes/TempestRemap/SB/f19/", \
           "/glade/campaign/cgd/amp/aherring/annelise_backup/TempestExtremes/TempestRemap/SB/ne30pg2/"/)

  runnam  = (/"f19", \
              "f09", \
              "ne30pg2", \
              "ne30pg3", \
              "ARCTIC", \
              "ARCTICGRIS", \
	      "ERA5", \
	      "MERRA2"/)

  nfiles = dimsizes(fname)

  VAR = "PRECT"
  factor = 1000.*secpday
  era5factor = 1000.*secpday/3600
  merra2factor = secpday

  labels = new( (/4*nfiles/), "string") 
  runnams = new( (/4*nfiles/), "string")

  pltNam = "temp_contours"

  cellfill = True

  griddir = (/"/glade/work/aherring/grids/uniform-res/f19/grids/", \
              "/glade/work/aherring/grids/uniform-res/f09/grids/", \
              "/glade/work/aherring/grids/uniform-res/ne30np4.pg2/grids/", \
              "/glade/work/aherring/grids/uniform-res/ne30np4.pg3/grids/", \
              "/glade/work/aherring/grids/var-res/ne0np4.ARCTIC.ne30x4/grids/", \
              "/glade/work/aherring/grids/var-res/ne0np4.ARCTICGRIS.ne30x8/grids/", \
	      "/glade/work/aherring/grids/SCRIP_files/", \
              "/glade/work/aherring/grids/SCRIP_files/"/)

  scrip = (/"fv1.9x2.5_141008.nc", \
            "fv0.9x1.25_scrip_141008.nc", \
            "ne30pg2_scrip_170608.nc", \
            "ne30pg3_scrip_170611.nc", \
            "ne0ARCTICne30x4_scrip_191212.nc", \
            "ne0ARCTICGRISne30x8_scrip_191209.nc", \
	    "ERA5_025_221025.nc", \
	    "MERRA2_scrip.nc"/)

;******************************************************
; load dates?
;******************************************************

  tmppath = "/glade/campaign/cgd/amp/aherring/archive/cam6_2_022.se_FHIST_f19_f19_mg17_900pes_200506_mg3-Nx5yrs/atm/hist/cam6_2_022.se_FHIST_f19_f19_mg17_900pes_200506_mg3-Nx5yrs.cam.h3.*"
   
;  sfiles = systemfunc ("ls " + tmppath)
;  f = addfiles (sfiles,"r")
;  ListSetType(f,"cat")
;  delete(sfiles)
;
;  dates = f[:]->date
;  delete(f)

;******************************************************
; create plot
;******************************************************
  wkstype = "pdf"
  wks = gsn_open_wks(wkstype,pltNam)               ; open workstation

  gsn_define_colormap(wks,"WhiteBlueGreenYellowRed")
  ;;gsn_define_colormap(wks,"BkBlAqGrYeOrReViWh200")
  ;;gsn_define_colormap(wks,"cmp_b2r")
  ;;gsn_reverse_colormap(wks)

  plot = new( (/4*nfiles/), "graphic")

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

;   res@mpProjection = "Orthographic"
;   res@mpCenterLatF      =   45.
;   res@mpCenterLonF      =  -45.
;
;   res@mpProjection = "LambertConformal";;"Orthographic";;"Mollweide"
;   res@mpLambertParallel1F = 85
;   res@mpLambertParallel2F = 58
;   res@mpLambertMeridianF  = -45
;   res@mpLimitMode       = "LatLon"
;   res@mpMinLonF = -56.4
;   res@mpMaxLonF = -30.
;   res@mpMinLatF = 59.4
;   res@mpMaxLatF = 85

   res@mpProjection = "LambertConformal"
   res@gsnMaskLambertConformal = True
   ;;res@gsnAddCyclic          = False
   ;;res@mpCenterLonF = 300.
   res@mpMinLonF =-90.;;-130.
   res@mpMaxLonF =  0.
   res@mpMinLatF = 45.;;20.
   res@mpMaxLatF = 85.

   resc = True
   resc@gsnMaximize = False
   resc@gsnDraw     = False      
   resc@gsnFrame    = False      
   resc@cnFillOn          = False
   resc@cnLinesOn         = True
   resc@cnLineLabelsOn    = False
   resc@cnInfoLabelOn     = False

   resp1 = True
   resp1 = resc
   resp1@cnLineColor = "blue";;"red"
   resp1@cnLineThicknessF = 1.5
   resp1@cnLevelSelectionMode = "ExplicitLevels"
   ;;resp1@cnLevels = (/1.,10000./)
   resp1@cnLevels = (/80.,120./)

   resp2 = True
   resp2 = resp1
   resp3 = True
   resp3 = resp1
   resp4 = True
   resp4 = resp1

;******************************************************
; begin loop
;******************************************************

 nens = 4
 myblobtc = new( (/nens,nfiles/), "integer" )

 ;;myblobtc(:,0) = (/17521,12786,3589,3934/)
 myblobtc(:,0) = (/17521,12786,3589,14079/)
 ;;myblobtc(:,1) = (/18706,26991,20008,12686/)
 myblobtc(:,1) = (/2035,26991,20008,12686/)
 myblobtc(:,2) = (/16135,21269,27244,1138/)
 myblobtc(:,3) = (/7479,25889,4256,6010/)
 ;;myblobtc(:,4) = (/28483,5177,17032,28953/)
 myblobtc(:,4) = (/28483,5177,22996,28953/)
 ;;myblobtc(:,5) = (/15542,44,24477,21693/)
 ;;myblobtc(:,5) = (/15542,44,24477,15875/)
 myblobtc(:,5) = (/15542,44,9151,15875/)

 do n = 0,nfiles-1

   era5data = False
   merra2data = False
   if (n.eq.nera5) then
     era5data = True
   else if (n.eq.nmerra2) then
     merra2data = True
   end if 
   end if

   print("load:"+subdir_scrip+fname0(n))
   f = addfile(subdir_scrip + fname0(n),"r")
   ;;var2 = f->av
   ;;xp2 = f->xp
   ;;yp2 = f->yp
   blobid = f->blobid
   ;;blobtc = f->blobtc

;if (n.lt.nfiles-2) then
;   myblobtc(:,n) = blobtc(:,kframe)
;end if
;print("blobid: " + blobid(:,kframe))
;print("blobtc: " + blobtc(:,kframe))
   delete(f)

;   do p=0,3
;     labels((n*4)+p) = tostring(dates(blobtc(p,kframe)))
;   end do

   if (era5data) then
     files = systemfunc ("ls " + era5dir + "*.nc")
   else if (merra2data) then
     files = systemfunc ("ls " + merra2dir + "*.nc")
   else
     files = systemfunc ("ls " + dir + fname(n) + rdir + fname(n) + fdate)
   end if
   end if

   nf = dimsizes(files)
   f  = addfile (files(0),"r")
   dims = getfiledimsizes(f)
   dimnames = getfiledimnames(f)

   if (any(dimnames.eq."lon")) then
     print("history fields on lat-lon grid")
     latlon = True
   else
     print("history fields on unstructured grid")
     latlon = False
   end if

   lat = f->lat
   lon = f->lon
   delete(f)

   ntime = 0
   do m = 0,nf-1

    print("load: " + files(m))
    f  = addfile (files(m),"r")
    dims = getfiledimsizes(f)
    dimnames = getfiledimnames(f)
    tind= ind(dimnames.eq."time")
    dtime = dims(tind)
    delete([/dims,dimnames/])

    if (m.eq.0) then
      ;;dims = dimsizes(blobtc)
      ;;nens = dims(0)
      ;;delete(dims)

      if (era5data) then
        vars = f->PRECT(0:nens-1,:,:)
      else if (merra2data) then
        vars = f->PRECTOT(0:nens-1,:,:)
      else
        if (latlon) then
          vars = f->PRECC(0:nens-1,:,:)
        else
          vars = f->PRECC(0:nens-1,:)
        end if   
      end if
      end if
      vars = 0.
      vars2 = vars
    end if

    np = 0
    do p = 0,nens-1
      ;;if ( (blobtc(p,kframe) .ge. ntime) .and. (blobtc(p,kframe) .lt. (ntime+dtime)) )
      ;;print("blobtc(p,kframe): " + blobtc(p,kframe) + " ntime: " + ntime + " ntime+dtime: " + (ntime+dtime)) 
      if ( (myblobtc(p,n) .ge. ntime) .and. (myblobtc(p,n) .lt. (ntime+dtime)) )
      print("myblobtc(p,n): " + myblobtc(p,n) + " ntime: " + ntime + " ntime+dtime: " + (ntime+dtime))
        dates = f->date
        ;;time = f->time
        ;;newtc= blobtc(p,kframe)-ntime
        newtc= myblobtc(p,n)-ntime
        ;;print("date: " + dates(newtc))
        ;;print("day: "  + time(newtc))

        labels((n*nens)+p) = tostring(dates(newtc))
        ;;delete([/dates,time/])
        delete(dates)

        if (era5data) then
          vars(p,:,:) = f->PRECT(newtc,:,:)
        else if (merra2data) then
          vars(p,:,:) = f->PRECTOT(newtc,:,:) ;;PRECTOTCORR
        else
          if (latlon) then
            vars(p,:,:) = f->PRECC(newtc,:,:)
            vartmp = f->PRECL(newtc,:,:)
            vars(p,:,:) = vars(p,:,:) + vartmp
            delete(vartmp)         

            vars2(p,:,:) = f->PSL(newtc,:,:)
          else
            vars(p,:) = f->PRECC(newtc,:)
            vartmp = f->PRECL(newtc,:)
            vars(p,:) = vars(p,:) + vartmp
            delete(vartmp)

            vars2(p,:) = f->PSL(newtc,:)
          end if
        end if
        end if

        ;;get blobs
        gfiles = systemfunc ("ls " + sdir(p) + fname(n) + "/" + "*.stitch.nc")
        print("load:"+gfiles(m))
        gf = addfile(gfiles(m),"r")
        delete(gfiles)

        if (p.eq.0) then
          vars3p1 = gf->object_id(newtc,:,:)
          if (n.eq.0) then
            latp1 = gf->lat
            lonp1 = gf->lon
          end if
        else if (p.eq.1) then
          vars3p2 = gf->object_id(newtc,:)
          if (n.eq.0) then
            latp2 = gf->lat
            lonp2 = gf->lon
          end if
        else if (p.eq.2) then
          vars3p3 = gf->object_id(newtc,:,:)
          if (n.eq.0) then
            latp3 = gf->lat
            lonp3 = gf->lon
          end if
        else if (p.eq.3) then
          vars3p4 = gf->object_id(newtc,:)
          if (n.eq.0) then
            latp4 = gf->lat
            lonp4 = gf->lon
          end if
        end if
        end if
        end if
        end if
        delete(gf)

        np = np + 1
      end if 
    end do

    ntime = ntime + dtime

    if (writencdf) then
      psl = f->PSL
      tmp = dim_avg_n(psl,0)
      tmp@_FillValue = default_fillvalue("float")
      delete(psl)
  
      if (m.eq.0) then
        dims = dimsizes(tmp)
        if (latlon) then
          sums = new( (/dims(0),dims(1)/), "float")
          sumsN = new( (/dims(0),dims(1)/), "float")
        else
          sums = new( (/dims(0)/), "float")
          sumsN = new( (/dims(0)/), "float")
        end if
        delete(dims)
  
        sums@_FillValue = default_fillvalue("float")
        sumsN@_FillValue = default_fillvalue("float")
        sums = 0
        sumsN = 0
      end if

      sums = sums + where(.not.ismissing(tmp),tmp,0.0)
      sumsN = sumsN + where(.not.ismissing(tmp),1.0,0.0)
      delete(tmp)
    end if

    delete(f)

    if (.not.writencdf) then
      ;;;no for climo.
      if (np.eq.nens) then
        continue
      end if
    end if

   end do 
   delete(files)

   if (era5data) then
     vars = vars*era5factor
   else if (merra2data) then
     vars = vars*merra2factor
   else
     vars = vars*factor
   end if
   end if

   if (writencdf) then
     sumsN = where(sumsN.lt.1.0,sumsN@_FillValue,sumsN)
     bar = sums/sumsN
     delete([/sums,sumsN/])

     fname_out = fname(n)+".climo.psl.nc"

     system("/bin/rm -f data/climo/" +fname_out)
     print("creating data/climo/" + fname_out)
     ncdf = addfile("data/climo/"+fname_out,"c")

     ncdf->bar = bar
     delete(ncdf)
   else if (loadncdf) then
     fname_in = fname(n)+".climo.psl.nc"

     print("loading ../data/climo/" + fname_in)
     ncdf = addfile("../data/climo/"+fname_in,"r")
     bar = ncdf->bar
     delete(ncdf)
   end if
   end if

   if (latlon) then
     vars!1 = "lat"
     vars&lat = lat
     vars&lat@units = "degrees_north"
     vars!2 = "lon"
     vars&lon = lon
     vars&lon@units = "degrees_east"

     ;;remove climo
     do p = 0,3
       vars2(p,:,:) = vars2(p,:,:) - bar
     end do
     vars2 = vars2/100.

     vars2!1 = "lat"
     vars2&lat = lat
     vars2&lat@units = "degrees_north"
     vars2!2 = "lon"
     vars2&lon = lon 
     vars2&lon@units = "degrees_east"
  
   else
     res@sfXArray = lon
     res@sfYArray = lat

     resc@sfXArray = lon
     resc@sfYArray = lat

     ;;remove climo
     do p = 0,3
       vars2(p,:) = vars2(p,:) - bar
     end do
     vars2 = vars2/100.

   end if
   delete([/lat,lon/])
   delete(bar)

   vars3p1 = where(vars3p1.gt.0,100,0)
   vars3p2 = where(vars3p2.gt.0,100,0)
   vars3p3 = where(vars3p3.gt.0,100,0)
   vars3p4 = where(vars3p4.gt.0,100,0)

   vars3p1!0 = "lat" 
   vars3p1&lat = latp1 
   vars3p1&lat@units = "degrees_north"
   vars3p1!1 = "lon"
   vars3p1&lon = lonp1 
   vars3p1&lon@units = "degrees_east"
   vars3p3!0 = "lat"
   vars3p3&lat = latp3
   vars3p3&lat@units = "degrees_north"
   vars3p3!1 = "lon"
   vars3p3&lon = lonp3
   vars3p3&lon@units = "degrees_east"

   resp2@sfXArray = lonp2
   resp2@sfYArray = latp2
   resp4@sfXArray = lonp4
   resp4@sfYArray = latp4

   print("max vars: " + max(vars))
   print("min vars: " + min(vars))

   print("max vars2: " + max(vars2))
   print("min vars2: " + min(vars2))

   print("max vars3p1: " + max(vars3p1))
   print("min vars3p1: " + min(vars3p1))
   print("max vars3p2: " + max(vars3p2))
   print("min vars3p2: " + min(vars3p2))
   print("max vars3p3: " + max(vars3p3))
   print("min vars3p3: " + min(vars3p3))
   print("max vars3p4: " + max(vars3p4))
   print("min vars3p4: " + min(vars3p4))

   res@cnLevelSelectionMode = "ExplicitLevels"
   res@cnLevels = (/0.5,1.,2.,5.,10.,15.,20.,30.,40.,50.,75.,100.,125.,150.,175.,200.,225.,250./)

   ;;resc@cnLineDrawOrder = "PostDraw"
   resc@cnLineColor = "black"
   resc@cnLineThicknessF = 0.5
   resc@gsnContourNegLineDashPattern = 1
   res@cnLevelSelectionMode = "ExplicitLevels"
   ;;set contour levels
   maxcn  = 100.
   mincn  =-100.
   cnlevs = 40.
   resc@cnLevelSelectionMode = "ManualLevels"
   resc@cnMinLevelValF  = mincn
   resc@cnMaxLevelValF  = maxcn
   resc@cnLevelSpacingF = (maxcn-mincn)/cnlevs

   if (cellfill) then
     res@cnFillMode = "CellFill"
     vf = addfile(griddir(n)+scrip(n),"r")
     res@sfXCellBounds = vf->grid_corner_lon
     res@sfYCellBounds = vf->grid_corner_lat
     delete(vf)
   else
     res@cnFillMode = "AreaFill"
   end if 

   delete(vars@standard_name)
   delete(vars@long_name)
   delete(vars@units)

   delete(vars2@long_name)
   delete(vars2@units)

   do p = 0,nens-1
    
     pid = n*nens + p

     vars@long_name = labels(pid)
     runnams(pid) = runnam(n)

     if (latlon) then
       plot(pid) = gsn_csm_contour_map(wks,vars(p,:,:),res)
       tmp = gsn_csm_contour(wks,vars2(p,:,:),resc)
       overlay(plot(pid),tmp)
       delete(tmp)
     else
       plot(pid) = gsn_csm_contour_map(wks,vars(p,:),res)
       tmp = gsn_csm_contour(wks,vars2(p,:),resc)
       overlay(plot(pid),tmp)
       delete(tmp)
     end if

   end do
   delete(vars)
   delete(vars2)
   
   ;;blobs
   pid = n*nens + 0
   tmp = gsn_csm_contour(wks,vars3p1,resp1)
   overlay(plot(pid),tmp)
   delete(tmp)
   pid = n*nens + 1  
   tmp = gsn_csm_contour(wks,vars3p2,resp2)
   overlay(plot(pid),tmp)       
   delete(tmp)
   pid = n*nens + 2  
   tmp = gsn_csm_contour(wks,vars3p3,resp3)
   overlay(plot(pid),tmp) 
   delete(tmp)
   pid = n*nens + 3  
   tmp = gsn_csm_contour(wks,vars3p4,resp4)
   overlay(plot(pid),tmp)
   delete(tmp)
   delete([/vars3p1,vars3p2,vars3p3,vars3p4/])

   if (.not.latlon) then
     delete([/res@sfXArray,res@sfYArray/])
     delete([/resc@sfXArray,resc@sfYArray/])
     delete([/resp2@sfXArray,resp2@sfYArray/])
     delete([/resp4@sfXArray,resp4@sfYArray/])
   end if

   if (cellfill) then
     delete([/res@sfXCellBounds,res@sfYCellBounds/])
   end if

 end do;;n

;;drawNDCGrid(wks)
;*****************************************************
; Plot panel
;*****************************************************

  resP                  = True                   ; modify the panel plot
  resP@gsnFrame         = False                  ; don't advance panel plot
  ;;resP@gsnPaperOrientation = "landscape"

  resP@gsnPanelFigureStrings = runnams
  resP@gsnPanelFigureStringsFontHeightF = 0.005
  resP@gsnPanelFigureStringsPerimOn     = True
  resP@gsnPanelFigureStringsBackgroundFillColor = "white"
  resP@amJust           = "TopLeft"

  resP@gsnPanelLabelBar = False
  resP@gsnPanelLabelBar = True
  resP@lbBoxSeparatorLinesOn= False
  resP@lbTitleString              = lbtitle
  resP@lbTitlePosition            = "Bottom"
  resP@lbTitleFontHeightF         = 0.015
  resP@lbLabelFontHeightF         = 0.01
  resP@pmLabelBarOrthogonalPosF   = -0.02

  resP@gsnPanelTop     = 0.95
  resP@gsnPanelBottom     = 0.05

  gsn_panel(wks,plot,(/nfiles,4/),resP)               ; now draw as one plot

frame(wks)

end

