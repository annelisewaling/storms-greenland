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
  p0 = 1000.d2
  secpday = 60.*60.*24;
  pi = 3.14159265358979
  g = 9.80616
  Re = 6371.22e3
  cp = 1004.
  Rd = 287.
;****************************************

  inttop = True
  intbot = True

  subdir_scrip = "../data/precip/"
     
;fname11 = (/"geolines_ESMF_PRECT_amean_f19.nc", \ 
;            "geolines_ESMF_PRECT_amean_f09.nc", \
;            "geolines_ESMF_PRECT_amean_ne30pg2.nc", \
;            "geolines_ESMF_PRECT_amean_ne30pg3.nc", \
;            "geolines_ESMF_PRECT_amean_ARCTIC.nc", \
;            "geolines_ESMF_PRECT_amean_ARCTICGRIS.nc", \
;            "geolines_PRECTOTCORR_PRECT_amean_ERA5.nc", \
;            "geolines_PRECTOTCORR_PRECT_amean_MERRA2.nc"/) 
;
;fname12 = (/"geolines_ESMF_PRECT_aint_f19.nc", \
;            "geolines_ESMF_PRECT_aint_f09.nc", \
;            "geolines_ESMF_PRECT_aint_ne30pg2.nc", \
;            "geolines_ESMF_PRECT_aint_ne30pg3.nc", \
;            "geolines_ESMF_PRECT_aint_ARCTIC.nc", \
;            "geolines_ESMF_PRECT_aint_ARCTICGRIS.nc", \
;            "geolines_PRECTOTCORR_PRECT_aint_ERA5.nc", \ 
;            "geolines_PRECTOTCORR_PRECT_aint_MERRA2.nc"/)
;
;fname21 = (/"geolines_PRECT_amean_ptile_f19.nc", \
;	    "geolines_PRECT_amean_ptile_f09.nc", \
;	    "geolines_PRECT_amean_ptile_ne30pg2.nc", \
;	    "geolines_PRECT_amean_ptile_ne30pg3.nc", \
;	    "geolines_PRECT_amean_ptile_ARCTIC.nc", \
;	    "geolines_PRECT_amean_ptile_ARCTICGRIS.nc", \
;	    "geolines_PRECT_amean_ptile_ERA5.nc", \
;            "geolines_PRECT_amean_ptile_MERRA2.nc"/)
;
;fname22 = (/"geolines_PRECT_aint_ptile_f19.nc", \
;	    "geolines_PRECT_aint_ptile_f09.nc", \
;	    "geolines_PRECT_aint_ptile_ne30pg2.nc", \
;	    "geolines_PRECT_aint_ptile_ne30pg3.nc", \
;	    "geolines_PRECT_aint_ptile_ARCTIC.nc", \
;	    "geolines_PRECT_aint_ptile_ARCTICGRIS.nc", \
;	    "geolines_PRECT_aint_ptile_ERA5.nc", \
;	    "geolines_PRECT_aint_ptile_MERRA2.nc"/)

fname11 = (/"geolines_PRECT_amean_pm1day_f19.nc", \
	    "geolines_PRECT_amean_pm1day_f09.nc", \
            "geolines_PRECT_amean_pm1day_ne30pg2.nc", \
            "geolines_PRECT_amean_pm1day_ne30pg3.nc", \
            "geolines_PRECT_amean_pm1day_ARCTIC.nc", \
            "geolines_PRECT_amean_pm1day_ARCTICGRIS.nc", \
            "geolines_PRECT_amean_pm1day_ERA5.nc", \
            "geolines_PRECT_amean_pm1day_MERRA2.nc"/)

fname12 = (/"geolines_PRECT_aint_pm1day_f19.nc", \
            "geolines_PRECT_aint_pm1day_f09.nc", \
            "geolines_PRECT_aint_pm1day_ne30pg2.nc", \
            "geolines_PRECT_aint_pm1day_ne30pg3.nc", \
            "geolines_PRECT_aint_pm1day_ARCTIC.nc", \
            "geolines_PRECT_aint_pm1day_ARCTICGRIS.nc", \
            "geolines_PRECT_aint_pm1day_ERA5.nc", \
            "geolines_PRECT_aint_pm1day_MERRA2.nc"/)

fname21 = (/"geolines_PRECT_amean_95ptile_f19.nc", \
	    "geolines_PRECT_amean_95ptile_f09.nc", \
            "geolines_PRECT_amean_95ptile_ne30pg2.nc", \
            "geolines_PRECT_amean_95ptile_ne30pg3.nc", \
            "geolines_PRECT_amean_95ptile_ARCTIC.nc", \
            "geolines_PRECT_amean_95ptile_ARCTICGRIS.nc", \
            "geolines_PRECT_amean_95ptile_ERA5.nc", \
            "geolines_PRECT_amean_95ptile_MERRA2.nc"/)

fname22 = (/"geolines_PRECT_aint_95ptile_f19.nc", \
            "geolines_PRECT_aint_95ptile_f09.nc", \
            "geolines_PRECT_aint_95ptile_ne30pg2.nc", \
            "geolines_PRECT_aint_95ptile_ne30pg3.nc", \
            "geolines_PRECT_aint_95ptile_ARCTIC.nc", \
            "geolines_PRECT_aint_95ptile_ARCTICGRIS.nc", \
            "geolines_PRECT_aint_95ptile_ERA5.nc", \
            "geolines_PRECT_aint_95ptile_MERRA2.nc"/)

nfiles = dimsizes(fname11)

labels = (/"f19", \
           "f09", \
           "ne30pg2", \
           "ne30pg3", \
           "ARCTIC", \
           "ARCTICGRIS", \
           "ERA5", \
           "MERRA2"/)

  colors = (/"blue","deepskyblue","green4","green","darkorchid4","magenta","black","black","grey49"/)

;******************************************************

  ;;meanframes = (/-2.,-1.75,-1.5,-1.25,-1.,-0.75,-0.50,-0.25,0.,0.25,0.50,0.75,1.,1.25,1.5,1.75,2./)
  ;;extframes = (/-1.5,-1.25,-1.,-0.75,-0.50,-0.25,0.,0.25,0.50,0.75,1.,1.25,1.5/)
  meanframes = (/-1.,-0.75,-0.50,-0.25,0.,0.25,0.50,0.75,1./)
  extframes = (/-1.,-0.75,-0.50,-0.25,0.,0.25,0.50,0.75,1./)
  dtcam = extframes(1)-extframes(0)

  xtime1 = meanframes
  xtime2 = extframes
  xtime3 = meanframes
  xtime4 = extframes

  if (inttop) then
    yaxis1 = "Cumulative Precip (mm)"
  else
    yaxis1 = "Precip Rate (mm/day)"
  end if  
  yaxis2 = yaxis1
  ;;yaxis2 = "Precip Rate (mm/day)"

  ymin1 = 0.
  ymax1 = 75.
  ymin2 = 50. 
  ymax2 = 150.;110.
  ymin2 = 0.    
  ymax2 = 190.

  if (inttop) then
    yaxis3 = "Area Int Cumulative Precip (Gt)"
  else
    yaxis3 = "Area Int Precip (Gt/day)"
  end if
  yaxis4 = yaxis3

  ymin3 =  0.
  ymax3 =  1.7
  ymin4 =  0.
  ymax4 =  8.5
  
  xaxis = "Days relative to max overlap with GrIS"
  
  xmina = -1.15
  xmaxa =  1.15
  
  xminb = -1.15
  xmaxb =  1.15

  pltNam = "temp_precip"

;******************************************************
; create plot
;******************************************************
  wkstype = "pdf"
  wks = gsn_open_wks(wkstype,pltNam)               ; open workstation

  plot = new( (/4/), "graphic")

   res             = True
   res@gsnMaximize = False
   res@gsnDraw     = False      ; Will draw later, after overlaying
   res@gsnFrame    = False      ; all plots

   res@vpWidthF            = 0.7           ; change aspect ratio of plot
   res@vpHeightF           = 0.3

   res@tmXMajorGridThicknessF = 2.
   res@tmYMajorGridThicknessF = 2.
   res@tmXMinorGridThicknessF = 1.
   res@tmYMinorGridThicknessF = 1.

   res@tmXBMajorThicknessF = 2.
   res@tmYRMajorThicknessF = 2.
   res@tmXTMajorThicknessF = 2.
   res@tmYLMajorThicknessF = 2.

   res@tmBorderThicknessF = 3.

   res@tmXBMajorOutwardLengthF = 0.
   res@tmYLMajorOutwardLengthF = 0.
   res@tmXBMinorOutwardLengthF = 0.
   res@tmYLMinorOutwardLengthF = 0.
   res@tmXTMajorOutwardLengthF = 0.
   res@tmYRMajorOutwardLengthF = 0.
   res@tmXTMinorOutwardLengthF = 0.
   res@tmYRMinorOutwardLengthF = 0.

   res@tiXAxisFontHeightF = 0.017
   res@tiYAxisFontHeightF = 0.017
   res@tmXBLabelFontHeightF = 0.015
   res@tmYLLabelFontHeightF = 0.017
   
   res@gsnYRefLine = 0.

;   res@tmGridDrawOrder = "PreDraw"
;   res@tmXMajorGrid= True
;   res@tmXMajorGridLineColor = "grey40"
;   res@tmXMajorGridThicknessF      = 1.
;   res@tmXMinorGrid= True
;   res@tmXMinorGridLineColor = "grey40"
;   res@tmXMinorGridThicknessF      = 1.
;   res@tmXMinorGridLineDashPattern = 2
;   res@tmYMajorGrid= True
;   res@tmYMajorGridLineColor = "grey40"
;   res@tmYMajorGridThicknessF      = 1.
;   res@tmYMinorGrid= True
;   res@tmYMinorGridLineColor = "grey40"
;   res@tmYMinorGridThicknessF      = 1.
;   res@tmYMinorGridLineDashPattern = 2
   
   ;;; transparent polygons
   gsres                = True
   gsres@gsFillOpacityF = 0.10;;0.2;;0.3
   gsres@tfPolyDrawOrder = "PreDraw"
 

;******************************************************
; begin loop
;******************************************************
     
  do n = 0,nfiles-1

   print("load:"+subdir_scrip+fname11(n))
   f = addfile(subdir_scrip + fname11(n),"r")
   var1 = f->av
   xp1 = f->xp
   yp1 = f->yp
   delete(f)

   print("load:"+subdir_scrip+fname12(n))
   f = addfile(subdir_scrip + fname12(n),"r")
   var3 = f->av
   xp3 = f->xp
   yp3 = f->yp
   delete(f)
   
   print("load:"+subdir_scrip+fname21(n))
   f = addfile(subdir_scrip + fname21(n),"r")
   var2 = f->av
   xp2 = f->xp
   yp2 = f->yp
   delete(f)

   print("load:"+subdir_scrip+fname22(n))
   f = addfile(subdir_scrip + fname22(n),"r")
   var4 = f->av
   xp4 = f->xp
   yp4 = f->yp
   delete(f)

   ;if (ismissing(var2(1))) then
   ;  var2(1) = 0.
   ;end if
   ;if (ismissing(var4(1))) then
   ;  var4(1) = 0.
   ;end if
   ;var2 = where(ismissing(var2),0.,var2)
   ;var4 = where(ismissing(var4),0.,var4)
   
   ;print(" precip: " + var2)
   ;print(" int precip: " + var4)
  
   if (inttop) then
       camcum = var1
       camcum = 0.
       n1 = dimsizes(camcum)

       up = yp1(1:n1)
       tmp= yp1(n1+1:2*n1)
       dn = tmp(::-1)
       delete(tmp)

       cumup1 = camcum
       cumup1 = 0.
       cumdn1 = cumup1
       do m = 1,n1-1
         camcum(m) = camcum(m-1) + dtcam*var1(m)
         cumup1(m) = cumup1(m-1) + dtcam*up(m)
         cumdn1(m) = cumdn1(m-1) + dtcam*dn(m)
       end do
       delete([/var1,up,dn/])
       var1 = camcum
       delete(camcum)

       camcum = var2
       camcum = 0.
       n2 = dimsizes(camcum)

       up = yp2(1:n2)
       tmp= yp2(n2+1:2*n2)
       dn = tmp(::-1)
       delete(tmp)

       cumup2 = camcum
       cumup2 = 0.
       cumdn2 = cumup2
       do m = 1,n2-1
         camcum(m) = camcum(m-1) + dtcam*var2(m)
         cumup2(m) = cumup2(m-1) + dtcam*up(m)
         cumdn2(m) = cumdn2(m-1) + dtcam*dn(m)
       end do
       delete(var2)
       var2 = camcum
       delete([/camcum,up,dn/])
   end if

   if (intbot) then
       camcum = var3
       camcum = 0.
       n3 = dimsizes(camcum)

       up = yp3(1:n3)
       tmp= yp3(n3+1:2*n3)
       dn = tmp(::-1)
       delete(tmp)

       cumup3 = camcum
       cumup3 = 0.
       cumdn3 = cumup3
       do m = 1,n3-1
         camcum(m) = camcum(m-1) + dtcam*var3(m)
         cumup3(m) = cumup3(m-1) + dtcam*up(m)
         cumdn3(m) = cumdn3(m-1) + dtcam*dn(m)
       end do
       delete([/var3,up,dn/])
       var3 = camcum
       delete(camcum)

       camcum = var4
       camcum = 0.
       n4 = dimsizes(camcum)

       up = yp4(1:n4)
       tmp= yp4(n4+1:2*n4)
       dn = tmp(::-1)
       delete(tmp)
     
       cumup4 = camcum
       cumup4 = 0.
       cumdn4 = cumup4
       do m = 1,n4-1
         camcum(m) = camcum(m-1) + dtcam*var4(m)
         cumup4(m) = cumup4(m-1) + dtcam*up(m)
         cumdn4(m) = cumdn4(m-1) + dtcam*dn(m)
       end do
       delete([/var4,up,dn/])
       var4 = camcum
       delete(camcum)
  end if

   res@xyDashPatterns = 0
   if (n.eq.nfiles-1) then
     res@xyDashPatterns = 16
   end if
   res@xyLineThicknessF  = 4
   res@xyLineColors = colors(n)

   if (n.eq.0) then
     res@tiYAxisString   = yaxis1
     res@trYMinF         = ymin1
     res@trYMaxF         = ymax1
     res@trXMinF         = xmina
     res@trXMaxF         = xmaxa
     plot(0)  = gsn_csm_xy(wks,xtime1,var1,res)

     res@tiYAxisString   = yaxis2
     res@trYMinF         = ymin2
     res@trYMaxF         = ymax2
     res@trXMinF         = xminb
     res@trXMaxF         = xmaxb
     plot(1)  = gsn_csm_xy(wks,xtime2,var2,res)

     res@tiXAxisString = xaxis
     res@tiYAxisString   = yaxis3
     res@trYMinF         = ymin3
     res@trYMaxF         = ymax3
     res@trXMinF         = xmina
     res@trXMaxF         = xmaxa
     plot(2)  = gsn_csm_xy(wks,xtime3,var3,res)

     res@tiXAxisString = xaxis
     res@tiYAxisString   = yaxis4
     res@trYMinF         = ymin4
     res@trYMaxF         = ymax4
     res@trXMinF         = xminb
     res@trXMaxF         = xmaxb
     plot(3)  = gsn_csm_xy(wks,xtime4,var4,res)

     delete(res@tiXAxisString)

   else
     tmp1 = gsn_csm_xy(wks,xtime1,var1,res)
     overlay(plot(0),tmp1) 
     delete(tmp1)

     tmp2 = gsn_csm_xy(wks,xtime2,var2,res)
     overlay(plot(1),tmp2) 
     delete(tmp2)

     tmp3 = gsn_csm_xy(wks,xtime3,var3,res)
     overlay(plot(2),tmp3) 
     delete(tmp3)

     tmp4 = gsn_csm_xy(wks,xtime4,var4,res)
     overlay(plot(3),tmp4) 
     delete(tmp4)
   end if
   
   delete([/var1,var2,var3,var4/])
   ;;now plot polygon shading
   gsres@gsFillColor    = colors(n)


   xp1 = xp1 - 1
   m1 = dimsizes(xp1)
   xxp1 = new( (/m1/), "float")
   do m = 0,m1-1
     xxp1(m) = xtime1(tointeger(xp1(m)))
   end do

   if (inttop) then
     yyp1 = new( (/m1/), "double")
     yyp1(0) = cumup1(0)
     yyp1(1:n1) = cumup1
     yyp1(n1+1:2*n1) = cumdn1(::-1)
   else
     yyp1 = yp1
   end if

   polystr = unique_string("hum")
   plot@$polystr$ = gsn_add_polygon(wks,plot(0),xxp1,yyp1,gsres)
   delete([/yyp1,xp1,yp1/])
    
   xp2 = xp2 - 1
   m2 = dimsizes(xp2)
   xxp2 = new( (/m2/), "float")
   do m = 0,m2-1
     xxp2(m) = xtime2(tointeger(xp2(m)))
   end do

   if (inttop) then
     yyp2 = new( (/m2/), "double")
     yyp2(0) = cumup2(0)
     yyp2(1:n2) = cumup2
     yyp2(n2+1:2*n2) = cumdn2(::-1)
   else
     yyp2 = yp2
   end if

   polystr = unique_string("hum")
   plot@$polystr$ = gsn_add_polygon(wks,plot(1),xxp2,yyp2,gsres)
   delete([/yyp2,xp2,yp2/])

   m3 = m1
   xxp3 = xxp1
   delete([/xxp1/])

   if (intbot) then
     yyp3 = new( (/m3/), "double")
     yyp3(0) = cumup3(0)
     yyp3(1:n3) = cumup3
     yyp3(n3+1:2*n3) = cumdn3(::-1)
   else
     yyp3 = yp3
   end if

   polystr = unique_string("hum")
   plot@$polystr$ = gsn_add_polygon(wks,plot(2),xxp3,yyp3,gsres)
   delete([/xxp3,yyp3,xp3,yp3/])

   m4 = m2
   xxp4 = xxp2
   delete([/xxp2/])

   if (intbot) then
     yyp4 = new( (/m4/), "double")
     yyp4(0) = cumup4(0)
     yyp4(1:n4) = cumup4
     yyp4(n4+1:2*n4) = cumdn4(::-1)
   else
     yyp4 = yp4
   end if

   polystr = unique_string("hum")
   plot@$polystr$ = gsn_add_polygon(wks,plot(3),xxp4,yyp4,gsres)
   delete([/xxp4,yyp4,xp4,yp4/])

 end do;;n


;*****************************************************
; Add legend
;*****************************************************
 txres = True
 txres@txFontHeightF = 0.010
 txres@txJust = "CenterLeft"

 resln             = True
 resln@gsLineThicknessF = res@xyLineThicknessF

 yy = (/0.74,0.725,0.71,0.695,0.68,0.665,0.65,0.635/)-0.07
 xx = 0.1125
 dx = 0.03
 do n = 0,nfiles-1
   resln@gsLineColor = colors(n)
   resln@gsLineDashPattern = 0
   if (n.eq.nfiles-1) then
     resln@gsLineDashPattern = 16
   end if
   gsn_polyline_ndc(wks,(/xx,xx+dx/),(/yy(n),yy(n)/),resln)
   gsn_text_ndc(wks,"  " + labels(n),xx+dx,yy(n),txres)
 end do

 txres@txFontHeightF = 0.015
 txres@txJust = "CenterCenter"

 delete(yy)
 yy = 0.705
 xx = 0.30
 gsn_text_ndc(wks,"Mean AR",xx,yy,txres)
 
 xx = 0.75
 gsn_text_ndc(wks,"95th Percentile AR",xx,yy,txres)

;;drawNDCGrid(wks)
;*****************************************************
; Plot panel
;*****************************************************

  resP                  = True                   ; modify the panel plot
  resP@gsnFrame         = False                  ; don't advance panel plot

  ;;resP@txString = plotTitle
  ;;resP@gsnPanelFigureStrings = runnams
  resP@gsnPanelFigureStrings= (/"a.","b.","c.","d."/)
  resP@gsnPanelFigureStringsFontHeightF = 0.01
  resP@gsnPanelFigureStringsPerimOn     = True
  resP@gsnPanelFigureStringsBackgroundFillColor = "white"
  resP@amJust           = "BottomRight"
  resP@gsnPanelLeft = 0.05
  resP@gsnPanelRight = 0.95
  ;;resP@gsnPanelBottom   = 0.1

  gsn_panel(wks,plot,(/2,2/),resP)

frame(wks)

end
