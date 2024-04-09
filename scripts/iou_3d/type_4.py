# NX 1953
# Journal created by Joker on Sun Jan 21 20:01:06 2024 中国标准时间

#
import sys
import math
import NXOpen
import NXOpen.Annotations
import NXOpen.Features
import NXOpen.GeometricUtilities
def without_bias(strip_section_stp_path, iou_3d_prt_path, springback_strip_line_path, prediction_line_path):
    try:
        theSession  = NXOpen.Session.GetSession()
        # ----------------------------------------------
        #   菜单：文件(F)->新建(N)...
        # ----------------------------------------------
        markId1 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        fileNew1 = theSession.Parts.FileNew()
        
        theSession.SetUndoMarkName(markId1, "新建 对话框")
        
        markId2 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "新建")
        
        theSession.DeleteUndoMark(markId2, None)
        
        markId3 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "新建")
        
        fileNew1.TemplateFileName = "model-plain-1-mm-template.prt"
        
        fileNew1.UseBlankTemplate = False
        
        fileNew1.ApplicationName = "ModelTemplate"
        
        fileNew1.Units = NXOpen.Part.Units.Millimeters
        
        fileNew1.RelationType = ""
        
        fileNew1.UsesMasterModel = "No"
        
        fileNew1.TemplateType = NXOpen.FileNewTemplateType.Item
        
        fileNew1.TemplatePresentationName = "模型"
        
        fileNew1.ItemType = ""
        
        fileNew1.Specialization = ""
        
        fileNew1.SetCanCreateAltrep(False)
        
        fileNew1.NewFileName = iou_3d_prt_path
        
        fileNew1.MasterFileName = ""
        
        fileNew1.MakeDisplayedPart = True
        
        fileNew1.DisplayPartOption = NXOpen.DisplayPartOption.AllowAdditional
        
        nXObject1 = fileNew1.Commit()
        
        workPart = theSession.Parts.Work
        displayPart = theSession.Parts.Display
        theSession.DeleteUndoMark(markId3, None)
        
        fileNew1.Destroy()
        
        theSession.ApplicationSwitchImmediate("UG_APP_MODELING")
        
        markId4 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects1 = [NXOpen.DisplayableObject.Null] * 8 
        datumAxis1 = workPart.Datums.FindObject("DATUM_CSYS(0) Y axis")
        objects1[0] = datumAxis1
        datumPlane1 = workPart.Datums.FindObject("DATUM_CSYS(0) XY plane")
        objects1[1] = datumPlane1
        datumPlane2 = workPart.Datums.FindObject("DATUM_CSYS(0) YZ plane")
        objects1[2] = datumPlane2
        datumAxis2 = workPart.Datums.FindObject("DATUM_CSYS(0) Z axis")
        objects1[3] = datumAxis2
        datumCsys1 = workPart.Features.FindObject("DATUM_CSYS(0)")
        point1 = datumCsys1.FindObject("POINT 1")
        objects1[4] = point1
        datumAxis3 = workPart.Datums.FindObject("DATUM_CSYS(0) X axis")
        objects1[5] = datumAxis3
        cartesianCoordinateSystem1 = datumCsys1.FindObject("CSYSTEM 1")
        objects1[6] = cartesianCoordinateSystem1
        datumPlane3 = workPart.Datums.FindObject("DATUM_CSYS(0) XZ plane")
        objects1[7] = datumPlane3
        theSession.DisplayManager.BlankObjects(objects1)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：文件(F)->导入(M)->STEP214...
        # ----------------------------------------------
        markId5 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        step214Importer1 = theSession.DexManager.CreateStep214Importer()
        
        step214Importer1.SimplifyGeometry = True
        
        step214Importer1.LayerDefault = 1
        
        step214Importer1.OutputFile = "G:\\PHD2\\Code\\e2e_ablation\\scripts\\iou_3d\\iou_3d_strip_section_type_4.prt"
        
        step214Importer1.SettingsFile = "D:\\NX1953\\NX1953\\step214ug\\step214ug.def"
        
        step214Importer1.ObjectTypes.ProductData = True
        
        step214Importer1.OutputFile = ""
        
        theSession.SetUndoMarkName(markId5, "导入 STEP214 文件 对话框")
        
        step214Importer1.SetMode(NXOpen.BaseImporter.Mode.NativeFileSystem)
        
        step214Importer1.OutputFile = "G:\\PHD2\\Code\\e2e_ablation\\scripts\\iou_3d\\iou_3d_strip_section_type_4.prt"
        
        step214Importer1.InputFile = strip_section_stp_path
        
        markId6 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "导入 STEP214 文件")
        
        theSession.DeleteUndoMark(markId6, None)
        
        markId7 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "导入 STEP214 文件")
        
        step214Importer1.FileOpenFlag = False
        
        step214Importer1.ProcessHoldFlag = True
        
        nXObject2 = step214Importer1.Commit()
        
        theSession.DeleteUndoMark(markId7, None)
        
        theSession.SetUndoMarkName(markId5, "导入 STEP214 文件")
        
        step214Importer1.Destroy()
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->7 抽取几何特征
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->关联复制(A)->抽取几何特征(E)...
        # ----------------------------------------------
        markId8 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        wavePointBuilder1 = workPart.Features.CreateWavePointBuilder(NXOpen.Features.Feature.Null)
        
        waveDatumBuilder1 = workPart.Features.CreateWaveDatumBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder1 = workPart.Features.CreateCompositeCurveBuilder(NXOpen.Features.Feature.Null)
        
        extractFaceBuilder1 = workPart.Features.CreateExtractFaceBuilder(NXOpen.Features.Feature.Null)
        
        mirrorBodyBuilder1 = workPart.Features.CreateMirrorBodyBuilder(NXOpen.Features.Feature.Null)
        
        waveSketchBuilder1 = workPart.Features.CreateWaveSketchBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder1.CurveFitData.Tolerance = 0.001
        
        compositeCurveBuilder1.CurveFitData.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder1.Section.SetAllowRefCrvs(False)
        
        extractFaceBuilder1.FaceOption = NXOpen.Features.ExtractFaceBuilder.FaceOptionType.AdjacentFaces
        
        compositeCurveBuilder1.Associative = False
        
        waveDatumBuilder1.ParentPart = NXOpen.Features.WaveDatumBuilder.ParentPartType.WorkPart
        
        wavePointBuilder1.ParentPart = NXOpen.Features.WavePointBuilder.ParentPartType.WorkPart
        
        extractFaceBuilder1.ParentPart = NXOpen.Features.ExtractFaceBuilder.ParentPartType.WorkPart
        
        mirrorBodyBuilder1.ParentPartType = NXOpen.Features.MirrorBodyBuilder.ParentPart.WorkPart
        
        compositeCurveBuilder1.ParentPart = NXOpen.Features.CompositeCurveBuilder.PartType.WorkPart
        
        waveSketchBuilder1.ParentPart = NXOpen.Features.WaveSketchBuilder.ParentPartType.WorkPart
        
        compositeCurveBuilder1.Associative = False
        
        theSession.SetUndoMarkName(markId8, "抽取几何特征 对话框")
        
        compositeCurveBuilder1.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder1.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder1.Section.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder1.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder1.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder1.Associative = False
        
        compositeCurveBuilder1.FixAtCurrentTimestamp = False
        
        compositeCurveBuilder1.HideOriginal = False
        
        compositeCurveBuilder1.InheritDisplayProperties = False
        
        compositeCurveBuilder1.Section.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.CurvesAndPoints)
        
        # ----------------------------------------------
        #   菜单：编辑(E)->选择(L)->全选(S)
        # ----------------------------------------------
        # Refer to the sample NXOpen application, Selection for "Select All" alternatives.
        markId9 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId10 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions1 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions1.SetSelectedFromInactive(False)
        
        curves1 = [NXOpen.IBaseCurve.Null] * 20 
        line1 = workPart.Lines.FindObject("ENTITY 3 4 1")
        curves1[0] = line1
        line2 = workPart.Lines.FindObject("ENTITY 3 3 1")
        curves1[1] = line2
        line3 = workPart.Lines.FindObject("ENTITY 3 2 1")
        curves1[2] = line3
        line4 = workPart.Lines.FindObject("ENTITY 3 1 1")
        curves1[3] = line4
        arc1 = workPart.Arcs.FindObject("ENTITY 5 16 1")
        curves1[4] = arc1
        arc2 = workPart.Arcs.FindObject("ENTITY 5 15 1")
        curves1[5] = arc2
        arc3 = workPart.Arcs.FindObject("ENTITY 5 14 1")
        curves1[6] = arc3
        arc4 = workPart.Arcs.FindObject("ENTITY 5 13 1")
        curves1[7] = arc4
        arc5 = workPart.Arcs.FindObject("ENTITY 5 12 1")
        curves1[8] = arc5
        arc6 = workPart.Arcs.FindObject("ENTITY 5 11 1")
        curves1[9] = arc6
        arc7 = workPart.Arcs.FindObject("ENTITY 5 10 1")
        curves1[10] = arc7
        arc8 = workPart.Arcs.FindObject("ENTITY 5 9 1")
        curves1[11] = arc8
        arc9 = workPart.Arcs.FindObject("ENTITY 5 8 1")
        curves1[12] = arc9
        arc10 = workPart.Arcs.FindObject("ENTITY 5 7 1")
        curves1[13] = arc10
        arc11 = workPart.Arcs.FindObject("ENTITY 5 6 1")
        curves1[14] = arc11
        arc12 = workPart.Arcs.FindObject("ENTITY 5 5 1")
        curves1[15] = arc12
        arc13 = workPart.Arcs.FindObject("ENTITY 5 4 1")
        curves1[16] = arc13
        arc14 = workPart.Arcs.FindObject("ENTITY 5 3 1")
        curves1[17] = arc14
        arc15 = workPart.Arcs.FindObject("ENTITY 5 2 1")
        curves1[18] = arc15
        arc16 = workPart.Arcs.FindObject("ENTITY 5 1 1")
        curves1[19] = arc16
        curveDumbRule1 = workPart.ScRuleFactory.CreateRuleBaseCurveDumb(curves1, selectionIntentRuleOptions1)
        
        selectionIntentRuleOptions1.Dispose()
        compositeCurveBuilder1.Section.AllowSelfIntersection(False)
        
        compositeCurveBuilder1.Section.AllowDegenerateCurves(False)
        
        rules1 = [None] * 1 
        rules1[0] = curveDumbRule1
        helpPoint1 = NXOpen.Point3d(0.0, 0.0, 0.0)
        compositeCurveBuilder1.Section.AddToSection(rules1, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint1, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId10, None)
        
        unit1 = workPart.UnitCollection.FindObject("MilliMeter")
        expression1 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        theSession.DeleteUndoMark(markId9, None)
        
        markId11 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        theSession.DeleteUndoMark(markId11, None)
        
        markId12 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        nXObject3 = compositeCurveBuilder1.Commit()
        
        theSession.DeleteUndoMark(markId12, None)
        
        theSession.SetUndoMarkName(markId8, "抽取几何特征")
        
        compositeCurveBuilder1.Destroy()
        
        waveDatumBuilder1.Destroy()
        
        wavePointBuilder1.Destroy()
        
        extractFaceBuilder1.Destroy()
        
        mirrorBodyBuilder1.Destroy()
        
        waveSketchBuilder1.Destroy()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression1)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        markId13 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects2 = [NXOpen.DisplayableObject.Null] * 20 
        compositeCurve1 = nXObject3
        arc17 = compositeCurve1.FindObject("CURVE 2 {5 (-5.4710880347518,4.9539266131459,0)}")
        objects2[0] = arc17
        arc18 = compositeCurve1.FindObject("CURVE 9 {5 (-4.3405111312096,1.3082355224489,0)}")
        objects2[1] = arc18
        arc19 = compositeCurve1.FindObject("CURVE 16 {5 (-4.3788990982709,-3.8513404120248,0)}")
        objects2[2] = arc19
        arc20 = compositeCurve1.FindObject("CURVE 5 {5 (-1.0985526453107,5.9739484737225,0)}")
        objects2[3] = arc20
        arc21 = compositeCurve1.FindObject("CURVE 12 {5 (-1.4294626721136,2.3897372482241,-0)}")
        objects2[4] = arc21
        line5 = compositeCurve1.FindObject("CURVE 19 {3 (-4.8790153392489,-7.8762724099349,0)}")
        objects2[5] = line5
        arc22 = compositeCurve1.FindObject("CURVE 8 {5 (-4.3788990982709,4.8374870357263,0)}")
        objects2[6] = arc22
        arc23 = compositeCurve1.FindObject("CURVE 15 {5 (-4.3405111312096,-0.3220888987474,0)}")
        objects2[7] = arc23
        arc24 = compositeCurve1.FindObject("CURVE 1 {5 (-5.7082046450936,0.4930733118508,0)}")
        objects2[8] = arc24
        arc25 = compositeCurve1.FindObject("CURVE 4 {5 (-0.1497274249929,8.3148811443444,0)}")
        objects2[9] = arc25
        arc26 = compositeCurve1.FindObject("CURVE 11 {5 (-1.110501427319,1.274729750782,0)}")
        objects2[10] = arc26
        arc27 = compositeCurve1.FindObject("CURVE 18 {5 (-4.070067140013,-9.2796047118725,0)}")
        objects2[11] = arc27
        arc28 = compositeCurve1.FindObject("CURVE 7 {5 (-3.2647295092912,8.7100247240138,0)}")
        objects2[12] = arc28
        line6 = compositeCurve1.FindObject("CURVE 14 {3 (-2.5660896449871,-0.0561158286467,0)}")
        objects2[13] = line6
        arc29 = compositeCurve1.FindObject("CURVE 3 {5 (-3.8109603721684,9.6629500582355,-0)}")
        objects2[14] = arc29
        line7 = compositeCurve1.FindObject("CURVE 10 {3 (-2.5660896449871,1.0422624523481,0)}")
        objects2[15] = line7
        line8 = compositeCurve1.FindObject("CURVE 17 {3 (-3.7978009634726,-7.6828549294017,0)}")
        objects2[16] = line8
        arc30 = compositeCurve1.FindObject("CURVE 6 {5 (-1.1834912602354,7.9437111279019,0)}")
        objects2[17] = arc30
        arc31 = compositeCurve1.FindObject("CURVE 13 {5 (-0.0537783872161,0.9751112188504,0)}")
        objects2[18] = arc31
        arc32 = compositeCurve1.FindObject("CURVE 20 {5 (-5.4710880347518,-3.9677799894444,0)}")
        objects2[19] = arc32
        theSession.DisplayManager.BlankObjects(objects2)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：编辑(E)->选择(L)->全选(S)
        # ----------------------------------------------
        # Refer to the sample NXOpen application, Selection for "Select All" alternatives.
        # ----------------------------------------------
        #   菜单：编辑(E)->删除(D)...
        # ----------------------------------------------
        markId14 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Delete")
        
        theSession.UpdateManager.ClearErrorList()
        
        markId15 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Delete")
        
        objects3 = [NXOpen.TaggedObject.Null] * 20 
        objects3[0] = line1
        objects3[1] = line2
        objects3[2] = line3
        objects3[3] = line4
        objects3[4] = arc1
        objects3[5] = arc2
        objects3[6] = arc3
        objects3[7] = arc4
        objects3[8] = arc5
        objects3[9] = arc6
        objects3[10] = arc7
        objects3[11] = arc8
        objects3[12] = arc9
        objects3[13] = arc10
        objects3[14] = arc11
        objects3[15] = arc12
        objects3[16] = arc13
        objects3[17] = arc14
        objects3[18] = arc15
        objects3[19] = arc16
        nErrs1 = theSession.UpdateManager.AddObjectsToDeleteList(objects3)
        
        id1 = theSession.NewestVisibleUndoMark
        
        nErrs2 = theSession.UpdateManager.DoUpdate(id1)
        
        theSession.DeleteUndoMark(markId14, None)
        
        # ----------------------------------------------
        #   菜单：编辑(E)->移动对象(O)...
        # ----------------------------------------------
        markId16 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        moveObjectBuilder1 = workPart.BaseFeatures.CreateMoveObjectBuilder(NXOpen.Features.MoveObject.Null)
        
        moveObjectBuilder1.TransformMotion.DistanceAngle.OrientXpress.AxisOption = NXOpen.GeometricUtilities.OrientXpressBuilder.Axis.Passive
        
        moveObjectBuilder1.TransformMotion.DistanceAngle.OrientXpress.PlaneOption = NXOpen.GeometricUtilities.OrientXpressBuilder.Plane.Passive
        
        moveObjectBuilder1.TransformMotion.AlongCurveAngle.AlongCurve.IsPercentUsed = True
        
        moveObjectBuilder1.TransformMotion.AlongCurveAngle.AlongCurve.Expression.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.AlongCurveAngle.AlongCurve.Expression.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.OrientXpress.AxisOption = NXOpen.GeometricUtilities.OrientXpressBuilder.Axis.Passive
        
        moveObjectBuilder1.TransformMotion.OrientXpress.PlaneOption = NXOpen.GeometricUtilities.OrientXpressBuilder.Plane.Passive
        
        moveObjectBuilder1.TransformMotion.DeltaEnum = NXOpen.GeometricUtilities.ModlMotion.Delta.ReferenceAcsWorkPart
        
        moveObjectBuilder1.TransformMotion.Option = NXOpen.GeometricUtilities.ModlMotion.Options.Angle
        
        moveObjectBuilder1.TransformMotion.DistanceValue.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.DistanceBetweenPointsDistance.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.RadialDistance.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.Angle.SetFormula("90")
        
        moveObjectBuilder1.TransformMotion.DistanceAngle.Distance.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.DistanceAngle.Angle.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.DeltaXc.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.DeltaYc.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.DeltaZc.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.AlongCurveAngle.AlongCurve.Expression.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.AlongCurveAngle.AlongCurveAngle.SetFormula("0")
        
        theSession.SetUndoMarkName(markId16, "移动对象 对话框")
        
        xform1 = workPart.Xforms.CreateXform(NXOpen.SmartObject.UpdateOption.WithinModeling, 1.0)
        
        cartesianCoordinateSystem2 = workPart.CoordinateSystems.CreateCoordinateSystem(xform1, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        moveObjectBuilder1.TransformMotion.ToCsys = cartesianCoordinateSystem2
        
        moveObjectBuilder1.TransformMotion.Angle.SetFormula("0")
        
        objects4 = [NXOpen.NXObject.Null] * 20 
        objects4[0] = arc24
        objects4[1] = arc17
        objects4[2] = arc29
        objects4[3] = arc25
        objects4[4] = arc20
        objects4[5] = arc30
        objects4[6] = arc28
        objects4[7] = arc22
        objects4[8] = arc18
        objects4[9] = line7
        objects4[10] = arc26
        objects4[11] = arc21
        objects4[12] = arc31
        objects4[13] = line6
        objects4[14] = arc23
        objects4[15] = arc19
        objects4[16] = line8
        objects4[17] = arc27
        objects4[18] = line5
        objects4[19] = arc32
        added1 = moveObjectBuilder1.ObjectToMoveObject.Add(objects4)
        
        markId17 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "起点")
        
        expression2 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        expression3 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        theSession.SetUndoMarkName(markId17, "矢量 对话框")
        
        # ----------------------------------------------
        #   对话开始 矢量
        # ----------------------------------------------
        origin1 = NXOpen.Point3d(0.0, 0.0, 0.0)
        vector1 = NXOpen.Vector3d(0.0, 1.0, 0.0)
        direction1 = workPart.Directions.CreateDirection(origin1, vector1, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        markId18 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "矢量")
        
        theSession.DeleteUndoMark(markId18, None)
        
        markId19 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "矢量")
        
        theSession.DeleteUndoMark(markId19, None)
        
        theSession.SetUndoMarkName(markId17, "矢量")
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression2)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression3)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        theSession.DeleteUndoMark(markId17, None)
        
        direction1.ProtectFromDelete()
        
        expression4 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        axis1 = workPart.Axes.CreateAxis(NXOpen.Point.Null, direction1, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        moveObjectBuilder1.TransformMotion.AngularAxis = axis1
        
        moveObjectBuilder1.TransformMotion.AngularAxis = axis1
        
        expression5 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        markId20 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "起点")
        
        expression6 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        expression7 = workPart.Expressions.CreateSystemExpressionWithUnits("p16_x=0.00000000000", unit1)
        
        expression8 = workPart.Expressions.CreateSystemExpressionWithUnits("p17_y=0.00000000000", unit1)
        
        expression9 = workPart.Expressions.CreateSystemExpressionWithUnits("p18_z=0.00000000000", unit1)
        
        expression10 = workPart.Expressions.CreateSystemExpressionWithUnits("p19_xdelta=0.00000000000", unit1)
        
        expression11 = workPart.Expressions.CreateSystemExpressionWithUnits("p20_ydelta=0.00000000000", unit1)
        
        expression12 = workPart.Expressions.CreateSystemExpressionWithUnits("p21_zdelta=0.00000000000", unit1)
        
        expression13 = workPart.Expressions.CreateSystemExpressionWithUnits("p22_radius=0.00000000000", unit1)
        
        unit2 = moveObjectBuilder1.TransformMotion.DistanceAngle.Angle.Units
        
        expression14 = workPart.Expressions.CreateSystemExpressionWithUnits("p23_angle=0.00000000000", unit2)
        
        expression15 = workPart.Expressions.CreateSystemExpressionWithUnits("p24_zdelta=0.00000000000", unit1)
        
        expression16 = workPart.Expressions.CreateSystemExpressionWithUnits("p25_radius=0.00000000000", unit1)
        
        expression17 = workPart.Expressions.CreateSystemExpressionWithUnits("p26_angle1=0.00000000000", unit2)
        
        expression18 = workPart.Expressions.CreateSystemExpressionWithUnits("p27_angle2=0.00000000000", unit2)
        
        expression19 = workPart.Expressions.CreateSystemExpressionWithUnits("p28_distance=0", unit1)
        
        expression20 = workPart.Expressions.CreateSystemExpressionWithUnits("p29_arclen=0", unit1)
        
        expression21 = workPart.Expressions.CreateSystemExpressionWithUnits("p30_percent=0", NXOpen.Unit.Null)
        
        expression7.SetFormula("0")
        
        expression8.SetFormula("0")
        
        expression9.SetFormula("0")
        
        expression10.SetFormula("0")
        
        expression11.SetFormula("0")
        
        expression12.SetFormula("0")
        
        expression13.SetFormula("0")
        
        expression14.SetFormula("0")
        
        expression15.SetFormula("0")
        
        expression16.SetFormula("0")
        
        expression17.SetFormula("0")
        
        expression18.SetFormula("0")
        
        expression19.SetFormula("0")
        
        expression21.SetFormula("100")
        
        expression20.SetFormula("0")
        
        theSession.SetUndoMarkName(markId20, "点 对话框")
        
        expression22 = workPart.Expressions.CreateSystemExpressionWithUnits("p31_x=0.00000000000", unit1)
        
        scalar1 = workPart.Scalars.CreateScalarExpression(expression22, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        expression23 = workPart.Expressions.CreateSystemExpressionWithUnits("p32_y=0.00000000000", unit1)
        
        scalar2 = workPart.Scalars.CreateScalarExpression(expression23, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        expression24 = workPart.Expressions.CreateSystemExpressionWithUnits("p33_z=0.00000000000", unit1)
        
        scalar3 = workPart.Scalars.CreateScalarExpression(expression24, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        point2 = workPart.Points.CreatePoint(scalar1, scalar2, scalar3, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        expression7.SetFormula("0.00000000000")
        
        expression8.SetFormula("0.00000000000")
        
        expression9.SetFormula("0.00000000000")
        
        expression10.SetFormula("0.00000000000")
        
        expression11.SetFormula("0.00000000000")
        
        expression12.SetFormula("0.00000000000")
        
        expression13.SetFormula("0.00000000000")
        
        expression14.SetFormula("0.00000000000")
        
        expression15.SetFormula("0.00000000000")
        
        expression16.SetFormula("0.00000000000")
        
        expression17.SetFormula("0.00000000000")
        
        expression18.SetFormula("0.00000000000")
        
        expression21.SetFormula("100.00000000000")
        
        # ----------------------------------------------
        #   对话开始 点
        # ----------------------------------------------
        markId21 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "点")
        
        theSession.DeleteUndoMark(markId21, None)
        
        markId22 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "点")
        
        expression7.RightHandSide = "0.00000000000"
        
        expression8.RightHandSide = "0.00000000000"
        
        expression9.RightHandSide = "0.00000000000"
        
        workPart.Points.DeletePoint(point2)
        
        expression25 = workPart.Expressions.CreateSystemExpressionWithUnits("p17_x=0.00000000000", unit1)
        
        scalar4 = workPart.Scalars.CreateScalarExpression(expression25, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        expression26 = workPart.Expressions.CreateSystemExpressionWithUnits("p18_y=0.00000000000", unit1)
        
        scalar5 = workPart.Scalars.CreateScalarExpression(expression26, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        expression27 = workPart.Expressions.CreateSystemExpressionWithUnits("p19_z=0.00000000000", unit1)
        
        scalar6 = workPart.Scalars.CreateScalarExpression(expression27, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        point3 = workPart.Points.CreatePoint(scalar4, scalar5, scalar6, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        theSession.DeleteUndoMark(markId22, None)
        
        theSession.SetUndoMarkName(markId20, "点")
        
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression7)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression8)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression9)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression10)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression11)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression12)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression13)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression14)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression15)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression16)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression17)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression18)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression19)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression20)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression21)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression6)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        theSession.DeleteUndoMark(markId20, None)
        
        scalar7 = workPart.Scalars.CreateScalarExpression(expression25, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        scalar8 = workPart.Scalars.CreateScalarExpression(expression26, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        scalar9 = workPart.Scalars.CreateScalarExpression(expression27, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        point4 = workPart.Points.CreatePoint(scalar7, scalar8, scalar9, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        point5 = axis1.Point
        
        axis1.Point = point3
        
        moveObjectBuilder1.TransformMotion.AngularAxis = axis1
        
        moveObjectBuilder1.TransformMotion.Angle.SetFormula("90")
        
        moveObjectBuilder1.TransformMotion.Angle.SetFormula("90")
        
        markId23 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "移动对象")
        
        theSession.DeleteUndoMark(markId23, None)
        
        markId24 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "移动对象")
        
        nXObject4 = moveObjectBuilder1.Commit()
        
        objects5 = moveObjectBuilder1.GetCommittedObjects()
        
        theSession.DeleteUndoMark(markId24, None)
        
        theSession.SetUndoMarkName(markId16, "移动对象")
        
        moveObjectBuilder1.Destroy()
        
        markId25 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "")
        
        nErrs3 = theSession.UpdateManager.DoUpdate(markId25)
        
        theSession.DeleteUndoMark(markId25, "")
        
        direction1.ReleaseDeleteProtection()
        
        workPart.Points.DeletePoint(point4)
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression5)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression4)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        # ----------------------------------------------
        #   菜单：文件(F)->导入(M)->文件中的点(L)...
        # ----------------------------------------------
        markId26 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Import Points from File")
        
        pointsFromFileBuilder1 = workPart.CreatePointsFromFileBuilder()
        
        pointsFromFileBuilder1.FileName = springback_strip_line_path
        
        pointsFromFileBuilder1.CoordinateOption = NXOpen.GeometricUtilities.PointsFromFileBuilder.Options.Absolute
        
        nXObject5 = pointsFromFileBuilder1.Commit()
        
        pointsFromFileBuilder1.Destroy()
        
        nErrs4 = theSession.UpdateManager.DoUpdate(markId26)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->10 拟合曲线
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->曲线(C)->拟合曲线(F)...
        # ----------------------------------------------
        markId27 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        fitCurveBuilder1 = workPart.Features.CreateFitCurveBuilder(NXOpen.Features.FitCurve.Null)
        
        fitCurveBuilder1.Tolerance = 0.001
        
        fitCurveBuilder1.TargetSourceType = NXOpen.Features.FitCurveBuilder.TargetSourceTypes.SpecifiedPoints
        
        fitCurveBuilder1.ProjectionDirectionOption = NXOpen.Features.FitCurveBuilder.ProjectionDirectionOptions.Normal
        
        fitCurveBuilder1.Radius.SetFormula("50")
        
        fitCurveBuilder1.Degree = 24
        
        fitCurveBuilder1.HasUniformSegments = True
        
        fitCurveBuilder1.Extender.StartValue.SetFormula("0")
        
        fitCurveBuilder1.Extender.EndValue.SetFormula("0")
        
        fitCurveBuilder1.RejectionThreshold.SetFormula("10")
        
        fitCurveBuilder1.IsAssociative = False
        
        theSession.SetUndoMarkName(markId27, "拟合曲线 对话框")
        
        markId28 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        theSession.DeleteUndoMark(markId28, None)
        
        markId29 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        # ----------------------------------------------
        #   菜单：编辑(E)->选择(L)->全选(S)
        # ----------------------------------------------
        # Refer to the sample NXOpen application, Selection for "Select All" alternatives.
        objects6 = [NXOpen.TaggedObject.Null] * 302 
        point6 = workPart.Points.FindObject("ENTITY 2 301 1")
        objects6[0] = point6
        point7 = workPart.Points.FindObject("ENTITY 2 300 1")
        objects6[1] = point7
        point8 = workPart.Points.FindObject("ENTITY 2 299 1")
        objects6[2] = point8
        point9 = workPart.Points.FindObject("ENTITY 2 298 1")
        objects6[3] = point9
        point10 = workPart.Points.FindObject("ENTITY 2 297 1")
        objects6[4] = point10
        point11 = workPart.Points.FindObject("ENTITY 2 296 1")
        objects6[5] = point11
        point12 = workPart.Points.FindObject("ENTITY 2 295 1")
        objects6[6] = point12
        point13 = workPart.Points.FindObject("ENTITY 2 294 1")
        objects6[7] = point13
        point14 = workPart.Points.FindObject("ENTITY 2 293 1")
        objects6[8] = point14
        point15 = workPart.Points.FindObject("ENTITY 2 292 1")
        objects6[9] = point15
        point16 = workPart.Points.FindObject("ENTITY 2 291 1")
        objects6[10] = point16
        point17 = workPart.Points.FindObject("ENTITY 2 290 1")
        objects6[11] = point17
        point18 = workPart.Points.FindObject("ENTITY 2 289 1")
        objects6[12] = point18
        point19 = workPart.Points.FindObject("ENTITY 2 288 1")
        objects6[13] = point19
        point20 = workPart.Points.FindObject("ENTITY 2 287 1")
        objects6[14] = point20
        point21 = workPart.Points.FindObject("ENTITY 2 286 1")
        objects6[15] = point21
        point22 = workPart.Points.FindObject("ENTITY 2 285 1")
        objects6[16] = point22
        point23 = workPart.Points.FindObject("ENTITY 2 284 1")
        objects6[17] = point23
        point24 = workPart.Points.FindObject("ENTITY 2 283 1")
        objects6[18] = point24
        point25 = workPart.Points.FindObject("ENTITY 2 282 1")
        objects6[19] = point25
        point26 = workPart.Points.FindObject("ENTITY 2 281 1")
        objects6[20] = point26
        point27 = workPart.Points.FindObject("ENTITY 2 280 1")
        objects6[21] = point27
        point28 = workPart.Points.FindObject("ENTITY 2 279 1")
        objects6[22] = point28
        point29 = workPart.Points.FindObject("ENTITY 2 278 1")
        objects6[23] = point29
        point30 = workPart.Points.FindObject("ENTITY 2 277 1")
        objects6[24] = point30
        point31 = workPart.Points.FindObject("ENTITY 2 276 1")
        objects6[25] = point31
        point32 = workPart.Points.FindObject("ENTITY 2 275 1")
        objects6[26] = point32
        point33 = workPart.Points.FindObject("ENTITY 2 274 1")
        objects6[27] = point33
        point34 = workPart.Points.FindObject("ENTITY 2 273 1")
        objects6[28] = point34
        point35 = workPart.Points.FindObject("ENTITY 2 272 1")
        objects6[29] = point35
        point36 = workPart.Points.FindObject("ENTITY 2 271 1")
        objects6[30] = point36
        point37 = workPart.Points.FindObject("ENTITY 2 270 1")
        objects6[31] = point37
        point38 = workPart.Points.FindObject("ENTITY 2 269 1")
        objects6[32] = point38
        point39 = workPart.Points.FindObject("ENTITY 2 268 1")
        objects6[33] = point39
        point40 = workPart.Points.FindObject("ENTITY 2 267 1")
        objects6[34] = point40
        point41 = workPart.Points.FindObject("ENTITY 2 266 1")
        objects6[35] = point41
        point42 = workPart.Points.FindObject("ENTITY 2 265 1")
        objects6[36] = point42
        point43 = workPart.Points.FindObject("ENTITY 2 264 1")
        objects6[37] = point43
        point44 = workPart.Points.FindObject("ENTITY 2 263 1")
        objects6[38] = point44
        point45 = workPart.Points.FindObject("ENTITY 2 262 1")
        objects6[39] = point45
        point46 = workPart.Points.FindObject("ENTITY 2 261 1")
        objects6[40] = point46
        point47 = workPart.Points.FindObject("ENTITY 2 260 1")
        objects6[41] = point47
        point48 = workPart.Points.FindObject("ENTITY 2 259 1")
        objects6[42] = point48
        point49 = workPart.Points.FindObject("ENTITY 2 258 1")
        objects6[43] = point49
        point50 = workPart.Points.FindObject("ENTITY 2 257 1")
        objects6[44] = point50
        point51 = workPart.Points.FindObject("ENTITY 2 256 1")
        objects6[45] = point51
        point52 = workPart.Points.FindObject("ENTITY 2 255 1")
        objects6[46] = point52
        point53 = workPart.Points.FindObject("ENTITY 2 254 1")
        objects6[47] = point53
        point54 = workPart.Points.FindObject("ENTITY 2 253 1")
        objects6[48] = point54
        point55 = workPart.Points.FindObject("ENTITY 2 252 1")
        objects6[49] = point55
        point56 = workPart.Points.FindObject("ENTITY 2 251 1")
        objects6[50] = point56
        point57 = workPart.Points.FindObject("ENTITY 2 250 1")
        objects6[51] = point57
        point58 = workPart.Points.FindObject("ENTITY 2 249 1")
        objects6[52] = point58
        point59 = workPart.Points.FindObject("ENTITY 2 248 1")
        objects6[53] = point59
        point60 = workPart.Points.FindObject("ENTITY 2 247 1")
        objects6[54] = point60
        point61 = workPart.Points.FindObject("ENTITY 2 246 1")
        objects6[55] = point61
        point62 = workPart.Points.FindObject("ENTITY 2 245 1")
        objects6[56] = point62
        point63 = workPart.Points.FindObject("ENTITY 2 244 1")
        objects6[57] = point63
        point64 = workPart.Points.FindObject("ENTITY 2 243 1")
        objects6[58] = point64
        point65 = workPart.Points.FindObject("ENTITY 2 242 1")
        objects6[59] = point65
        point66 = workPart.Points.FindObject("ENTITY 2 241 1")
        objects6[60] = point66
        point67 = workPart.Points.FindObject("ENTITY 2 240 1")
        objects6[61] = point67
        point68 = workPart.Points.FindObject("ENTITY 2 239 1")
        objects6[62] = point68
        point69 = workPart.Points.FindObject("ENTITY 2 238 1")
        objects6[63] = point69
        point70 = workPart.Points.FindObject("ENTITY 2 237 1")
        objects6[64] = point70
        point71 = workPart.Points.FindObject("ENTITY 2 236 1")
        objects6[65] = point71
        point72 = workPart.Points.FindObject("ENTITY 2 235 1")
        objects6[66] = point72
        point73 = workPart.Points.FindObject("ENTITY 2 234 1")
        objects6[67] = point73
        point74 = workPart.Points.FindObject("ENTITY 2 233 1")
        objects6[68] = point74
        point75 = workPart.Points.FindObject("ENTITY 2 232 1")
        objects6[69] = point75
        point76 = workPart.Points.FindObject("ENTITY 2 231 1")
        objects6[70] = point76
        point77 = workPart.Points.FindObject("ENTITY 2 230 1")
        objects6[71] = point77
        point78 = workPart.Points.FindObject("ENTITY 2 229 1")
        objects6[72] = point78
        point79 = workPart.Points.FindObject("ENTITY 2 228 1")
        objects6[73] = point79
        point80 = workPart.Points.FindObject("ENTITY 2 227 1")
        objects6[74] = point80
        point81 = workPart.Points.FindObject("ENTITY 2 226 1")
        objects6[75] = point81
        point82 = workPart.Points.FindObject("ENTITY 2 225 1")
        objects6[76] = point82
        point83 = workPart.Points.FindObject("ENTITY 2 224 1")
        objects6[77] = point83
        point84 = workPart.Points.FindObject("ENTITY 2 223 1")
        objects6[78] = point84
        point85 = workPart.Points.FindObject("ENTITY 2 222 1")
        objects6[79] = point85
        point86 = workPart.Points.FindObject("ENTITY 2 221 1")
        objects6[80] = point86
        point87 = workPart.Points.FindObject("ENTITY 2 220 1")
        objects6[81] = point87
        point88 = workPart.Points.FindObject("ENTITY 2 219 1")
        objects6[82] = point88
        point89 = workPart.Points.FindObject("ENTITY 2 218 1")
        objects6[83] = point89
        point90 = workPart.Points.FindObject("ENTITY 2 217 1")
        objects6[84] = point90
        point91 = workPart.Points.FindObject("ENTITY 2 216 1")
        objects6[85] = point91
        point92 = workPart.Points.FindObject("ENTITY 2 215 1")
        objects6[86] = point92
        point93 = workPart.Points.FindObject("ENTITY 2 214 1")
        objects6[87] = point93
        point94 = workPart.Points.FindObject("ENTITY 2 213 1")
        objects6[88] = point94
        point95 = workPart.Points.FindObject("ENTITY 2 212 1")
        objects6[89] = point95
        point96 = workPart.Points.FindObject("ENTITY 2 211 1")
        objects6[90] = point96
        point97 = workPart.Points.FindObject("ENTITY 2 210 1")
        objects6[91] = point97
        point98 = workPart.Points.FindObject("ENTITY 2 209 1")
        objects6[92] = point98
        point99 = workPart.Points.FindObject("ENTITY 2 208 1")
        objects6[93] = point99
        point100 = workPart.Points.FindObject("ENTITY 2 207 1")
        objects6[94] = point100
        point101 = workPart.Points.FindObject("ENTITY 2 206 1")
        objects6[95] = point101
        point102 = workPart.Points.FindObject("ENTITY 2 205 1")
        objects6[96] = point102
        point103 = workPart.Points.FindObject("ENTITY 2 204 1")
        objects6[97] = point103
        point104 = workPart.Points.FindObject("ENTITY 2 203 1")
        objects6[98] = point104
        point105 = workPart.Points.FindObject("ENTITY 2 202 1")
        objects6[99] = point105
        point106 = workPart.Points.FindObject("ENTITY 2 201 1")
        objects6[100] = point106
        point107 = workPart.Points.FindObject("ENTITY 2 200 1")
        objects6[101] = point107
        point108 = workPart.Points.FindObject("ENTITY 2 199 1")
        objects6[102] = point108
        point109 = workPart.Points.FindObject("ENTITY 2 198 1")
        objects6[103] = point109
        point110 = workPart.Points.FindObject("ENTITY 2 197 1")
        objects6[104] = point110
        point111 = workPart.Points.FindObject("ENTITY 2 196 1")
        objects6[105] = point111
        point112 = workPart.Points.FindObject("ENTITY 2 195 1")
        objects6[106] = point112
        point113 = workPart.Points.FindObject("ENTITY 2 194 1")
        objects6[107] = point113
        point114 = workPart.Points.FindObject("ENTITY 2 193 1")
        objects6[108] = point114
        point115 = workPart.Points.FindObject("ENTITY 2 192 1")
        objects6[109] = point115
        point116 = workPart.Points.FindObject("ENTITY 2 191 1")
        objects6[110] = point116
        point117 = workPart.Points.FindObject("ENTITY 2 190 1")
        objects6[111] = point117
        point118 = workPart.Points.FindObject("ENTITY 2 189 1")
        objects6[112] = point118
        point119 = workPart.Points.FindObject("ENTITY 2 188 1")
        objects6[113] = point119
        point120 = workPart.Points.FindObject("ENTITY 2 187 1")
        objects6[114] = point120
        point121 = workPart.Points.FindObject("ENTITY 2 186 1")
        objects6[115] = point121
        point122 = workPart.Points.FindObject("ENTITY 2 185 1")
        objects6[116] = point122
        point123 = workPart.Points.FindObject("ENTITY 2 184 1")
        objects6[117] = point123
        point124 = workPart.Points.FindObject("ENTITY 2 183 1")
        objects6[118] = point124
        point125 = workPart.Points.FindObject("ENTITY 2 182 1")
        objects6[119] = point125
        point126 = workPart.Points.FindObject("ENTITY 2 181 1")
        objects6[120] = point126
        point127 = workPart.Points.FindObject("ENTITY 2 180 1")
        objects6[121] = point127
        point128 = workPart.Points.FindObject("ENTITY 2 179 1")
        objects6[122] = point128
        point129 = workPart.Points.FindObject("ENTITY 2 178 1")
        objects6[123] = point129
        point130 = workPart.Points.FindObject("ENTITY 2 177 1")
        objects6[124] = point130
        point131 = workPart.Points.FindObject("ENTITY 2 176 1")
        objects6[125] = point131
        point132 = workPart.Points.FindObject("ENTITY 2 175 1")
        objects6[126] = point132
        point133 = workPart.Points.FindObject("ENTITY 2 174 1")
        objects6[127] = point133
        point134 = workPart.Points.FindObject("ENTITY 2 173 1")
        objects6[128] = point134
        point135 = workPart.Points.FindObject("ENTITY 2 172 1")
        objects6[129] = point135
        point136 = workPart.Points.FindObject("ENTITY 2 171 1")
        objects6[130] = point136
        point137 = workPart.Points.FindObject("ENTITY 2 170 1")
        objects6[131] = point137
        point138 = workPart.Points.FindObject("ENTITY 2 169 1")
        objects6[132] = point138
        point139 = workPart.Points.FindObject("ENTITY 2 168 1")
        objects6[133] = point139
        point140 = workPart.Points.FindObject("ENTITY 2 167 1")
        objects6[134] = point140
        point141 = workPart.Points.FindObject("ENTITY 2 166 1")
        objects6[135] = point141
        point142 = workPart.Points.FindObject("ENTITY 2 165 1")
        objects6[136] = point142
        point143 = workPart.Points.FindObject("ENTITY 2 164 1")
        objects6[137] = point143
        point144 = workPart.Points.FindObject("ENTITY 2 163 1")
        objects6[138] = point144
        point145 = workPart.Points.FindObject("ENTITY 2 162 1")
        objects6[139] = point145
        point146 = workPart.Points.FindObject("ENTITY 2 161 1")
        objects6[140] = point146
        point147 = workPart.Points.FindObject("ENTITY 2 160 1")
        objects6[141] = point147
        point148 = workPart.Points.FindObject("ENTITY 2 159 1")
        objects6[142] = point148
        point149 = workPart.Points.FindObject("ENTITY 2 158 1")
        objects6[143] = point149
        point150 = workPart.Points.FindObject("ENTITY 2 157 1")
        objects6[144] = point150
        point151 = workPart.Points.FindObject("ENTITY 2 156 1")
        objects6[145] = point151
        point152 = workPart.Points.FindObject("ENTITY 2 155 1")
        objects6[146] = point152
        point153 = workPart.Points.FindObject("ENTITY 2 154 1")
        objects6[147] = point153
        point154 = workPart.Points.FindObject("ENTITY 2 153 1")
        objects6[148] = point154
        point155 = workPart.Points.FindObject("ENTITY 2 152 1")
        objects6[149] = point155
        point156 = workPart.Points.FindObject("ENTITY 2 151 1")
        objects6[150] = point156
        point157 = workPart.Points.FindObject("ENTITY 2 150 1")
        objects6[151] = point157
        point158 = workPart.Points.FindObject("ENTITY 2 149 1")
        objects6[152] = point158
        point159 = workPart.Points.FindObject("ENTITY 2 148 1")
        objects6[153] = point159
        point160 = workPart.Points.FindObject("ENTITY 2 147 1")
        objects6[154] = point160
        point161 = workPart.Points.FindObject("ENTITY 2 146 1")
        objects6[155] = point161
        point162 = workPart.Points.FindObject("ENTITY 2 145 1")
        objects6[156] = point162
        point163 = workPart.Points.FindObject("ENTITY 2 144 1")
        objects6[157] = point163
        point164 = workPart.Points.FindObject("ENTITY 2 143 1")
        objects6[158] = point164
        point165 = workPart.Points.FindObject("ENTITY 2 142 1")
        objects6[159] = point165
        point166 = workPart.Points.FindObject("ENTITY 2 141 1")
        objects6[160] = point166
        point167 = workPart.Points.FindObject("ENTITY 2 140 1")
        objects6[161] = point167
        point168 = workPart.Points.FindObject("ENTITY 2 139 1")
        objects6[162] = point168
        point169 = workPart.Points.FindObject("ENTITY 2 138 1")
        objects6[163] = point169
        point170 = workPart.Points.FindObject("ENTITY 2 137 1")
        objects6[164] = point170
        point171 = workPart.Points.FindObject("ENTITY 2 136 1")
        objects6[165] = point171
        point172 = workPart.Points.FindObject("ENTITY 2 135 1")
        objects6[166] = point172
        point173 = workPart.Points.FindObject("ENTITY 2 134 1")
        objects6[167] = point173
        point174 = workPart.Points.FindObject("ENTITY 2 133 1")
        objects6[168] = point174
        point175 = workPart.Points.FindObject("ENTITY 2 132 1")
        objects6[169] = point175
        point176 = workPart.Points.FindObject("ENTITY 2 131 1")
        objects6[170] = point176
        point177 = workPart.Points.FindObject("ENTITY 2 130 1")
        objects6[171] = point177
        point178 = workPart.Points.FindObject("ENTITY 2 129 1")
        objects6[172] = point178
        point179 = workPart.Points.FindObject("ENTITY 2 128 1")
        objects6[173] = point179
        point180 = workPart.Points.FindObject("ENTITY 2 127 1")
        objects6[174] = point180
        point181 = workPart.Points.FindObject("ENTITY 2 126 1")
        objects6[175] = point181
        point182 = workPart.Points.FindObject("ENTITY 2 125 1")
        objects6[176] = point182
        point183 = workPart.Points.FindObject("ENTITY 2 124 1")
        objects6[177] = point183
        point184 = workPart.Points.FindObject("ENTITY 2 123 1")
        objects6[178] = point184
        point185 = workPart.Points.FindObject("ENTITY 2 122 1")
        objects6[179] = point185
        point186 = workPart.Points.FindObject("ENTITY 2 121 1")
        objects6[180] = point186
        point187 = workPart.Points.FindObject("ENTITY 2 120 1")
        objects6[181] = point187
        point188 = workPart.Points.FindObject("ENTITY 2 119 1")
        objects6[182] = point188
        point189 = workPart.Points.FindObject("ENTITY 2 118 1")
        objects6[183] = point189
        point190 = workPart.Points.FindObject("ENTITY 2 117 1")
        objects6[184] = point190
        point191 = workPart.Points.FindObject("ENTITY 2 116 1")
        objects6[185] = point191
        point192 = workPart.Points.FindObject("ENTITY 2 115 1")
        objects6[186] = point192
        point193 = workPart.Points.FindObject("ENTITY 2 114 1")
        objects6[187] = point193
        point194 = workPart.Points.FindObject("ENTITY 2 113 1")
        objects6[188] = point194
        point195 = workPart.Points.FindObject("ENTITY 2 112 1")
        objects6[189] = point195
        point196 = workPart.Points.FindObject("ENTITY 2 111 1")
        objects6[190] = point196
        point197 = workPart.Points.FindObject("ENTITY 2 110 1")
        objects6[191] = point197
        point198 = workPart.Points.FindObject("ENTITY 2 109 1")
        objects6[192] = point198
        point199 = workPart.Points.FindObject("ENTITY 2 108 1")
        objects6[193] = point199
        point200 = workPart.Points.FindObject("ENTITY 2 107 1")
        objects6[194] = point200
        point201 = workPart.Points.FindObject("ENTITY 2 106 1")
        objects6[195] = point201
        point202 = workPart.Points.FindObject("ENTITY 2 105 1")
        objects6[196] = point202
        point203 = workPart.Points.FindObject("ENTITY 2 104 1")
        objects6[197] = point203
        point204 = workPart.Points.FindObject("ENTITY 2 103 1")
        objects6[198] = point204
        point205 = workPart.Points.FindObject("ENTITY 2 102 1")
        objects6[199] = point205
        point206 = workPart.Points.FindObject("ENTITY 2 101 1")
        objects6[200] = point206
        point207 = workPart.Points.FindObject("ENTITY 2 100 1")
        objects6[201] = point207
        point208 = workPart.Points.FindObject("ENTITY 2 99 1")
        objects6[202] = point208
        point209 = workPart.Points.FindObject("ENTITY 2 98 1")
        objects6[203] = point209
        point210 = workPart.Points.FindObject("ENTITY 2 97 1")
        objects6[204] = point210
        point211 = workPart.Points.FindObject("ENTITY 2 96 1")
        objects6[205] = point211
        point212 = workPart.Points.FindObject("ENTITY 2 95 1")
        objects6[206] = point212
        point213 = workPart.Points.FindObject("ENTITY 2 94 1")
        objects6[207] = point213
        point214 = workPart.Points.FindObject("ENTITY 2 93 1")
        objects6[208] = point214
        point215 = workPart.Points.FindObject("ENTITY 2 92 1")
        objects6[209] = point215
        point216 = workPart.Points.FindObject("ENTITY 2 91 1")
        objects6[210] = point216
        point217 = workPart.Points.FindObject("ENTITY 2 90 1")
        objects6[211] = point217
        point218 = workPart.Points.FindObject("ENTITY 2 89 1")
        objects6[212] = point218
        point219 = workPart.Points.FindObject("ENTITY 2 88 1")
        objects6[213] = point219
        point220 = workPart.Points.FindObject("ENTITY 2 87 1")
        objects6[214] = point220
        point221 = workPart.Points.FindObject("ENTITY 2 86 1")
        objects6[215] = point221
        point222 = workPart.Points.FindObject("ENTITY 2 85 1")
        objects6[216] = point222
        point223 = workPart.Points.FindObject("ENTITY 2 84 1")
        objects6[217] = point223
        point224 = workPart.Points.FindObject("ENTITY 2 83 1")
        objects6[218] = point224
        point225 = workPart.Points.FindObject("ENTITY 2 82 1")
        objects6[219] = point225
        point226 = workPart.Points.FindObject("ENTITY 2 81 1")
        objects6[220] = point226
        point227 = workPart.Points.FindObject("ENTITY 2 80 1")
        objects6[221] = point227
        point228 = workPart.Points.FindObject("ENTITY 2 79 1")
        objects6[222] = point228
        point229 = workPart.Points.FindObject("ENTITY 2 78 1")
        objects6[223] = point229
        point230 = workPart.Points.FindObject("ENTITY 2 77 1")
        objects6[224] = point230
        point231 = workPart.Points.FindObject("ENTITY 2 76 1")
        objects6[225] = point231
        point232 = workPart.Points.FindObject("ENTITY 2 75 1")
        objects6[226] = point232
        point233 = workPart.Points.FindObject("ENTITY 2 74 1")
        objects6[227] = point233
        point234 = workPart.Points.FindObject("ENTITY 2 73 1")
        objects6[228] = point234
        point235 = workPart.Points.FindObject("ENTITY 2 72 1")
        objects6[229] = point235
        point236 = workPart.Points.FindObject("ENTITY 2 71 1")
        objects6[230] = point236
        point237 = workPart.Points.FindObject("ENTITY 2 70 1")
        objects6[231] = point237
        point238 = workPart.Points.FindObject("ENTITY 2 69 1")
        objects6[232] = point238
        point239 = workPart.Points.FindObject("ENTITY 2 68 1")
        objects6[233] = point239
        point240 = workPart.Points.FindObject("ENTITY 2 67 1")
        objects6[234] = point240
        point241 = workPart.Points.FindObject("ENTITY 2 66 1")
        objects6[235] = point241
        point242 = workPart.Points.FindObject("ENTITY 2 65 1")
        objects6[236] = point242
        point243 = workPart.Points.FindObject("ENTITY 2 64 1")
        objects6[237] = point243
        point244 = workPart.Points.FindObject("ENTITY 2 63 1")
        objects6[238] = point244
        point245 = workPart.Points.FindObject("ENTITY 2 62 1")
        objects6[239] = point245
        point246 = workPart.Points.FindObject("ENTITY 2 61 1")
        objects6[240] = point246
        point247 = workPart.Points.FindObject("ENTITY 2 60 1")
        objects6[241] = point247
        point248 = workPart.Points.FindObject("ENTITY 2 59 1")
        objects6[242] = point248
        point249 = workPart.Points.FindObject("ENTITY 2 58 1")
        objects6[243] = point249
        point250 = workPart.Points.FindObject("ENTITY 2 57 1")
        objects6[244] = point250
        point251 = workPart.Points.FindObject("ENTITY 2 56 1")
        objects6[245] = point251
        point252 = workPart.Points.FindObject("ENTITY 2 55 1")
        objects6[246] = point252
        point253 = workPart.Points.FindObject("ENTITY 2 54 1")
        objects6[247] = point253
        point254 = workPart.Points.FindObject("ENTITY 2 53 1")
        objects6[248] = point254
        point255 = workPart.Points.FindObject("ENTITY 2 52 1")
        objects6[249] = point255
        point256 = workPart.Points.FindObject("ENTITY 2 51 1")
        objects6[250] = point256
        point257 = workPart.Points.FindObject("ENTITY 2 50 1")
        objects6[251] = point257
        point258 = workPart.Points.FindObject("ENTITY 2 49 1")
        objects6[252] = point258
        point259 = workPart.Points.FindObject("ENTITY 2 48 1")
        objects6[253] = point259
        point260 = workPart.Points.FindObject("ENTITY 2 47 1")
        objects6[254] = point260
        point261 = workPart.Points.FindObject("ENTITY 2 46 1")
        objects6[255] = point261
        point262 = workPart.Points.FindObject("ENTITY 2 45 1")
        objects6[256] = point262
        point263 = workPart.Points.FindObject("ENTITY 2 44 1")
        objects6[257] = point263
        point264 = workPart.Points.FindObject("ENTITY 2 43 1")
        objects6[258] = point264
        point265 = workPart.Points.FindObject("ENTITY 2 42 1")
        objects6[259] = point265
        point266 = workPart.Points.FindObject("ENTITY 2 41 1")
        objects6[260] = point266
        point267 = workPart.Points.FindObject("ENTITY 2 40 1")
        objects6[261] = point267
        point268 = workPart.Points.FindObject("ENTITY 2 39 1")
        objects6[262] = point268
        point269 = workPart.Points.FindObject("ENTITY 2 38 1")
        objects6[263] = point269
        point270 = workPart.Points.FindObject("ENTITY 2 37 1")
        objects6[264] = point270
        point271 = workPart.Points.FindObject("ENTITY 2 36 1")
        objects6[265] = point271
        point272 = workPart.Points.FindObject("ENTITY 2 35 1")
        objects6[266] = point272
        point273 = workPart.Points.FindObject("ENTITY 2 34 1")
        objects6[267] = point273
        point274 = workPart.Points.FindObject("ENTITY 2 33 1")
        objects6[268] = point274
        point275 = workPart.Points.FindObject("ENTITY 2 32 1")
        objects6[269] = point275
        point276 = workPart.Points.FindObject("ENTITY 2 31 1")
        objects6[270] = point276
        point277 = workPart.Points.FindObject("ENTITY 2 30 1")
        objects6[271] = point277
        point278 = workPart.Points.FindObject("ENTITY 2 29 1")
        objects6[272] = point278
        point279 = workPart.Points.FindObject("ENTITY 2 28 1")
        objects6[273] = point279
        point280 = workPart.Points.FindObject("ENTITY 2 27 1")
        objects6[274] = point280
        point281 = workPart.Points.FindObject("ENTITY 2 26 1")
        objects6[275] = point281
        point282 = workPart.Points.FindObject("ENTITY 2 25 1")
        objects6[276] = point282
        point283 = workPart.Points.FindObject("ENTITY 2 24 1")
        objects6[277] = point283
        point284 = workPart.Points.FindObject("ENTITY 2 23 1")
        objects6[278] = point284
        point285 = workPart.Points.FindObject("ENTITY 2 22 1")
        objects6[279] = point285
        point286 = workPart.Points.FindObject("ENTITY 2 21 1")
        objects6[280] = point286
        point287 = workPart.Points.FindObject("ENTITY 2 20 1")
        objects6[281] = point287
        point288 = workPart.Points.FindObject("ENTITY 2 19 1")
        objects6[282] = point288
        point289 = workPart.Points.FindObject("ENTITY 2 18 1")
        objects6[283] = point289
        point290 = workPart.Points.FindObject("ENTITY 2 17 1")
        objects6[284] = point290
        point291 = workPart.Points.FindObject("ENTITY 2 16 1")
        objects6[285] = point291
        point292 = workPart.Points.FindObject("ENTITY 2 15 1")
        objects6[286] = point292
        point293 = workPart.Points.FindObject("ENTITY 2 14 1")
        objects6[287] = point293
        point294 = workPart.Points.FindObject("ENTITY 2 13 1")
        objects6[288] = point294
        point295 = workPart.Points.FindObject("ENTITY 2 12 1")
        objects6[289] = point295
        point296 = workPart.Points.FindObject("ENTITY 2 11 1")
        objects6[290] = point296
        point297 = workPart.Points.FindObject("ENTITY 2 10 1")
        objects6[291] = point297
        point298 = workPart.Points.FindObject("ENTITY 2 9 1")
        objects6[292] = point298
        point299 = workPart.Points.FindObject("ENTITY 2 8 1")
        objects6[293] = point299
        point300 = workPart.Points.FindObject("ENTITY 2 7 1")
        objects6[294] = point300
        point301 = workPart.Points.FindObject("ENTITY 2 6 1")
        objects6[295] = point301
        point302 = workPart.Points.FindObject("ENTITY 2 5 1")
        objects6[296] = point302
        point303 = workPart.Points.FindObject("ENTITY 2 4 1")
        objects6[297] = point303
        point304 = workPart.Points.FindObject("ENTITY 2 3 1")
        objects6[298] = point304
        point305 = workPart.Points.FindObject("ENTITY 2 2 1")
        objects6[299] = point305
        point306 = workPart.Points.FindObject("ENTITY 2 1 1")
        objects6[300] = point306
        group1 = nXObject5
        objects6[301] = group1
        added2 = fitCurveBuilder1.Target.Add(objects6)
        
        geometricConstraintData1 = fitCurveBuilder1.ConstraintManager.FindItem(0)
        
        point307 = geometricConstraintData1.Point
        
        geometricConstraintData2 = fitCurveBuilder1.ConstraintManager.FindItem(1)
        
        point308 = geometricConstraintData2.Point
        
        theSession.SetUndoMarkName(markId29, "拟合曲线 - 选择")
        
        theSession.SetUndoMarkVisibility(markId29, None, NXOpen.Session.MarkVisibility.Visible)
        
        theSession.SetUndoMarkVisibility(markId27, None, NXOpen.Session.MarkVisibility.Invisible)
        
        markId30 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        theSession.DeleteUndoMark(markId30, None)
        
        markId31 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        fitCurveBuilder1.HasReversedDirection = True
        
        theSession.SetUndoMarkName(markId31, "拟合曲线 - 反向")
        
        theSession.SetUndoMarkVisibility(markId31, None, NXOpen.Session.MarkVisibility.Visible)
        
        theSession.SetUndoMarkVisibility(markId27, None, NXOpen.Session.MarkVisibility.Invisible)
        
        markId32 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        markId33 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        theSession.DeleteUndoMark(markId33, None)
        
        markId34 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        nXObject6 = fitCurveBuilder1.Commit()
        
        theSession.DeleteUndoMark(markId34, None)
        
        theSession.SetUndoMarkName(markId27, "拟合曲线")
        
        fitCurveBuilder1.Destroy()
        
        theSession.SetUndoMarkVisibility(markId27, None, NXOpen.Session.MarkVisibility.Visible)
        
        theSession.DeleteUndoMark(markId31, None)
        
        theSession.DeleteUndoMark(markId29, None)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->5 抽取几何特征
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->关联复制(A)->抽取几何特征(E)...
        # ----------------------------------------------
        markId35 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        wavePointBuilder2 = workPart.Features.CreateWavePointBuilder(NXOpen.Features.Feature.Null)
        
        waveDatumBuilder2 = workPart.Features.CreateWaveDatumBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder2 = workPart.Features.CreateCompositeCurveBuilder(NXOpen.Features.Feature.Null)
        
        extractFaceBuilder2 = workPart.Features.CreateExtractFaceBuilder(NXOpen.Features.Feature.Null)
        
        mirrorBodyBuilder2 = workPart.Features.CreateMirrorBodyBuilder(NXOpen.Features.Feature.Null)
        
        waveSketchBuilder2 = workPart.Features.CreateWaveSketchBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder2.CurveFitData.Tolerance = 0.001
        
        compositeCurveBuilder2.CurveFitData.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder2.Section.SetAllowRefCrvs(False)
        
        extractFaceBuilder2.FaceOption = NXOpen.Features.ExtractFaceBuilder.FaceOptionType.AdjacentFaces
        
        compositeCurveBuilder2.Associative = False
        
        waveDatumBuilder2.ParentPart = NXOpen.Features.WaveDatumBuilder.ParentPartType.WorkPart
        
        wavePointBuilder2.ParentPart = NXOpen.Features.WavePointBuilder.ParentPartType.WorkPart
        
        extractFaceBuilder2.ParentPart = NXOpen.Features.ExtractFaceBuilder.ParentPartType.WorkPart
        
        mirrorBodyBuilder2.ParentPartType = NXOpen.Features.MirrorBodyBuilder.ParentPart.WorkPart
        
        compositeCurveBuilder2.ParentPart = NXOpen.Features.CompositeCurveBuilder.PartType.WorkPart
        
        waveSketchBuilder2.ParentPart = NXOpen.Features.WaveSketchBuilder.ParentPartType.WorkPart
        
        compositeCurveBuilder2.Associative = False
        
        theSession.SetUndoMarkName(markId35, "抽取几何特征 对话框")
        
        compositeCurveBuilder2.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder2.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder2.Section.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder2.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder2.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder2.Associative = False
        
        compositeCurveBuilder2.FixAtCurrentTimestamp = False
        
        compositeCurveBuilder2.HideOriginal = False
        
        compositeCurveBuilder2.InheritDisplayProperties = False
        
        compositeCurveBuilder2.Section.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.CurvesAndPoints)
        
        markId36 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId37 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions2 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions2.SetSelectedFromInactive(False)
        
        curves2 = [NXOpen.IBaseCurve.Null] * 1 
        spline1 = workPart.Splines.FindObject("ENTITY 9 1 1")
        curves2[0] = spline1
        curveDumbRule2 = workPart.ScRuleFactory.CreateRuleBaseCurveDumb(curves2, selectionIntentRuleOptions2)
        
        selectionIntentRuleOptions2.Dispose()
        compositeCurveBuilder2.Section.AllowSelfIntersection(False)
        
        compositeCurveBuilder2.Section.AllowDegenerateCurves(False)
        
        rules2 = [None] * 1 
        rules2[0] = curveDumbRule2
        helpPoint2 = NXOpen.Point3d(8.0855494375583277, 0.052325682805135193, -0.032092608049883731)
        compositeCurveBuilder2.Section.AddToSection(rules2, spline1, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint2, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId37, None)
        
        theSession.DeleteUndoMark(markId36, None)
        
        markId38 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        theSession.DeleteUndoMark(markId38, None)
        
        markId39 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        nXObject7 = compositeCurveBuilder2.Commit()
        
        theSession.DeleteUndoMark(markId39, None)
        
        theSession.SetUndoMarkName(markId35, "抽取几何特征")
        
        compositeCurveBuilder2.Destroy()
        
        waveDatumBuilder2.Destroy()
        
        wavePointBuilder2.Destroy()
        
        extractFaceBuilder2.Destroy()
        
        mirrorBodyBuilder2.Destroy()
        
        waveSketchBuilder2.Destroy()
        
        markId40 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects7 = [NXOpen.DisplayableObject.Null] * 1 
        compositeCurve2 = nXObject7
        spline2 = compositeCurve2.FindObject("CURVE 1")
        objects7[0] = spline2
        theSession.DisplayManager.BlankObjects(objects7)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：编辑(E)->选择(L)->全选(S)
        # ----------------------------------------------
        # Refer to the sample NXOpen application, Selection for "Select All" alternatives.
        # ----------------------------------------------
        #   菜单：编辑(E)->删除(D)...
        # ----------------------------------------------
        markId41 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Delete")
        
        theSession.UpdateManager.ClearErrorList()
        
        markId42 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Delete")
        
        objects8 = [NXOpen.TaggedObject.Null] * 303 
        objects8[0] = point6
        objects8[1] = point7
        objects8[2] = point8
        objects8[3] = point9
        objects8[4] = point10
        objects8[5] = point11
        objects8[6] = point12
        objects8[7] = point13
        objects8[8] = point14
        objects8[9] = point15
        objects8[10] = point16
        objects8[11] = point17
        objects8[12] = point18
        objects8[13] = point19
        objects8[14] = point20
        objects8[15] = point21
        objects8[16] = point22
        objects8[17] = point23
        objects8[18] = point24
        objects8[19] = point25
        objects8[20] = point26
        objects8[21] = point27
        objects8[22] = point28
        objects8[23] = point29
        objects8[24] = point30
        objects8[25] = point31
        objects8[26] = point32
        objects8[27] = point33
        objects8[28] = point34
        objects8[29] = point35
        objects8[30] = point36
        objects8[31] = point37
        objects8[32] = point38
        objects8[33] = point39
        objects8[34] = point40
        objects8[35] = point41
        objects8[36] = point42
        objects8[37] = point43
        objects8[38] = point44
        objects8[39] = point45
        objects8[40] = point271
        objects8[41] = point272
        objects8[42] = point273
        objects8[43] = point274
        objects8[44] = point275
        objects8[45] = point276
        objects8[46] = point277
        objects8[47] = point278
        objects8[48] = point279
        objects8[49] = point280
        objects8[50] = point281
        objects8[51] = point282
        objects8[52] = point283
        objects8[53] = point284
        objects8[54] = point285
        objects8[55] = point286
        objects8[56] = point287
        objects8[57] = point288
        objects8[58] = point289
        objects8[59] = point290
        objects8[60] = point291
        objects8[61] = point292
        objects8[62] = point293
        objects8[63] = point294
        objects8[64] = point295
        objects8[65] = point296
        objects8[66] = point297
        objects8[67] = point298
        objects8[68] = point299
        objects8[69] = point300
        objects8[70] = point301
        objects8[71] = point302
        objects8[72] = point303
        objects8[73] = point304
        objects8[74] = point305
        objects8[75] = point306
        objects8[76] = group1
        objects8[77] = point162
        objects8[78] = point163
        objects8[79] = point164
        objects8[80] = point165
        objects8[81] = point166
        objects8[82] = point167
        objects8[83] = point168
        objects8[84] = point169
        objects8[85] = point170
        objects8[86] = point171
        objects8[87] = point172
        objects8[88] = point173
        objects8[89] = point174
        objects8[90] = point175
        objects8[91] = point176
        objects8[92] = point177
        objects8[93] = point178
        objects8[94] = point179
        objects8[95] = point180
        objects8[96] = point181
        objects8[97] = point182
        objects8[98] = point183
        objects8[99] = point184
        objects8[100] = point185
        objects8[101] = point186
        objects8[102] = point187
        objects8[103] = point188
        objects8[104] = point189
        objects8[105] = point190
        objects8[106] = point191
        objects8[107] = point192
        objects8[108] = point193
        objects8[109] = point194
        objects8[110] = point195
        objects8[111] = point196
        objects8[112] = point197
        objects8[113] = point198
        objects8[114] = point199
        objects8[115] = point200
        objects8[116] = point201
        objects8[117] = point202
        objects8[118] = point203
        objects8[119] = point204
        objects8[120] = point205
        objects8[121] = point206
        objects8[122] = point207
        objects8[123] = point73
        objects8[124] = point74
        objects8[125] = point75
        objects8[126] = point76
        objects8[127] = point77
        objects8[128] = point78
        objects8[129] = point79
        objects8[130] = point80
        objects8[131] = point81
        objects8[132] = point82
        objects8[133] = point83
        objects8[134] = point84
        objects8[135] = point85
        objects8[136] = point86
        objects8[137] = point87
        objects8[138] = point88
        objects8[139] = point89
        objects8[140] = point90
        objects8[141] = point91
        objects8[142] = point92
        objects8[143] = point93
        objects8[144] = point94
        objects8[145] = point95
        objects8[146] = point96
        objects8[147] = point97
        objects8[148] = point98
        objects8[149] = point99
        objects8[150] = point100
        objects8[151] = point101
        objects8[152] = point102
        objects8[153] = point103
        objects8[154] = point104
        objects8[155] = point105
        objects8[156] = point106
        objects8[157] = point107
        objects8[158] = point108
        objects8[159] = point109
        objects8[160] = point110
        objects8[161] = point111
        objects8[162] = point112
        objects8[163] = point113
        objects8[164] = point114
        objects8[165] = point115
        objects8[166] = point116
        objects8[167] = point117
        objects8[168] = point118
        objects8[169] = point119
        objects8[170] = point120
        objects8[171] = point121
        objects8[172] = point122
        objects8[173] = point123
        objects8[174] = point124
        objects8[175] = point125
        objects8[176] = point126
        objects8[177] = point46
        objects8[178] = point47
        objects8[179] = point48
        objects8[180] = point49
        objects8[181] = point50
        objects8[182] = point51
        objects8[183] = point52
        objects8[184] = point53
        objects8[185] = point54
        objects8[186] = point55
        objects8[187] = spline1
        objects8[188] = point208
        objects8[189] = point209
        objects8[190] = point210
        objects8[191] = point211
        objects8[192] = point212
        objects8[193] = point213
        objects8[194] = point214
        objects8[195] = point215
        objects8[196] = point216
        objects8[197] = point217
        objects8[198] = point218
        objects8[199] = point219
        objects8[200] = point220
        objects8[201] = point221
        objects8[202] = point222
        objects8[203] = point223
        objects8[204] = point224
        objects8[205] = point225
        objects8[206] = point226
        objects8[207] = point227
        objects8[208] = point228
        objects8[209] = point229
        objects8[210] = point230
        objects8[211] = point231
        objects8[212] = point232
        objects8[213] = point233
        objects8[214] = point234
        objects8[215] = point235
        objects8[216] = point236
        objects8[217] = point237
        objects8[218] = point238
        objects8[219] = point239
        objects8[220] = point240
        objects8[221] = point241
        objects8[222] = point242
        objects8[223] = point243
        objects8[224] = point244
        objects8[225] = point245
        objects8[226] = point246
        objects8[227] = point247
        objects8[228] = point248
        objects8[229] = point249
        objects8[230] = point250
        objects8[231] = point251
        objects8[232] = point252
        objects8[233] = point253
        objects8[234] = point254
        objects8[235] = point255
        objects8[236] = point256
        objects8[237] = point257
        objects8[238] = point258
        objects8[239] = point259
        objects8[240] = point260
        objects8[241] = point261
        objects8[242] = point262
        objects8[243] = point263
        objects8[244] = point264
        objects8[245] = point265
        objects8[246] = point266
        objects8[247] = point267
        objects8[248] = point268
        objects8[249] = point269
        objects8[250] = point270
        objects8[251] = point127
        objects8[252] = point128
        objects8[253] = point129
        objects8[254] = point130
        objects8[255] = point131
        objects8[256] = point132
        objects8[257] = point133
        objects8[258] = point134
        objects8[259] = point135
        objects8[260] = point136
        objects8[261] = point137
        objects8[262] = point138
        objects8[263] = point139
        objects8[264] = point140
        objects8[265] = point141
        objects8[266] = point142
        objects8[267] = point143
        objects8[268] = point144
        objects8[269] = point145
        objects8[270] = point146
        objects8[271] = point147
        objects8[272] = point148
        objects8[273] = point149
        objects8[274] = point150
        objects8[275] = point151
        objects8[276] = point152
        objects8[277] = point153
        objects8[278] = point154
        objects8[279] = point155
        objects8[280] = point156
        objects8[281] = point157
        objects8[282] = point158
        objects8[283] = point159
        objects8[284] = point160
        objects8[285] = point161
        objects8[286] = point56
        objects8[287] = point57
        objects8[288] = point58
        objects8[289] = point59
        objects8[290] = point60
        objects8[291] = point61
        objects8[292] = point62
        objects8[293] = point63
        objects8[294] = point64
        objects8[295] = point65
        objects8[296] = point66
        objects8[297] = point67
        objects8[298] = point68
        objects8[299] = point69
        objects8[300] = point70
        objects8[301] = point71
        objects8[302] = point72
        nErrs5 = theSession.UpdateManager.AddObjectsToDeleteList(objects8)
        
        id2 = theSession.NewestVisibleUndoMark
        
        nErrs6 = theSession.UpdateManager.DoUpdate(id2)
        
        theSession.DeleteUndoMark(markId41, None)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->10 扫掠
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->扫掠(W)->扫掠(S)...
        # ----------------------------------------------
        markId43 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        sweptBuilder1 = workPart.Features.CreateSweptBuilder(NXOpen.Features.Swept.Null)
        
        expression28 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        expression29 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        expression30 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        sweptBuilder1.G0Tolerance = 0.001
        
        sweptBuilder1.G1Tolerance = 0.050000000000000003
        
        sweptBuilder1.OrientationMethod.AngularLaw.Value.SetFormula("0")
        
        sweptBuilder1.OrientationMethod.AngularLaw.StartValue.SetFormula("0")
        
        sweptBuilder1.OrientationMethod.AngularLaw.EndValue.SetFormula("0")
        
        sweptBuilder1.ScalingMethod.AreaLaw.Value.SetFormula("1")
        
        sweptBuilder1.ScalingMethod.AreaLaw.StartValue.SetFormula("1")
        
        sweptBuilder1.ScalingMethod.AreaLaw.EndValue.SetFormula("1")
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.Value.SetFormula("1")
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.StartValue.SetFormula("1")
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.EndValue.SetFormula("1")
        
        theSession.SetUndoMarkName(markId43, "扫掠 对话框")
        
        sweptBuilder1.Spine.DistanceTolerance = 0.001
        
        sweptBuilder1.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder1.AlignmentMethod.AlignCurve.DistanceTolerance = 0.001
        
        sweptBuilder1.AlignmentMethod.AlignCurve.ChainingTolerance = 0.00095
        
        sweptBuilder1.OrientationMethod.OrientationCurve.DistanceTolerance = 0.001
        
        sweptBuilder1.OrientationMethod.OrientationCurve.ChainingTolerance = 0.00095
        
        sweptBuilder1.OrientationMethod.AngularLaw.AlongSpineData.Spine.DistanceTolerance = 0.001
        
        sweptBuilder1.OrientationMethod.AngularLaw.AlongSpineData.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder1.OrientationMethod.AngularLaw.LawCurve.DistanceTolerance = 0.001
        
        sweptBuilder1.OrientationMethod.AngularLaw.LawCurve.ChainingTolerance = 0.00095
        
        sweptBuilder1.ScalingMethod.ScalingCurve.DistanceTolerance = 0.001
        
        sweptBuilder1.ScalingMethod.ScalingCurve.ChainingTolerance = 0.00095
        
        sweptBuilder1.ScalingMethod.AreaLaw.AlongSpineData.Spine.DistanceTolerance = 0.001
        
        sweptBuilder1.ScalingMethod.AreaLaw.AlongSpineData.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder1.ScalingMethod.AreaLaw.LawCurve.DistanceTolerance = 0.001
        
        sweptBuilder1.ScalingMethod.AreaLaw.LawCurve.ChainingTolerance = 0.00095
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.AlongSpineData.Spine.DistanceTolerance = 0.001
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.AlongSpineData.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.LawCurve.DistanceTolerance = 0.001
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.LawCurve.ChainingTolerance = 0.00095
        
        sweptBuilder1.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.AlignmentMethod.AlignCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.OrientationMethod.OrientationCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.OrientationMethod.AngularLaw.AlongSpineData.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.OrientationMethod.AngularLaw.LawCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.ScalingMethod.ScalingCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.ScalingMethod.AreaLaw.AlongSpineData.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.ScalingMethod.AreaLaw.LawCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.AlongSpineData.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.LawCurve.AngleTolerance = 0.050000000000000003
        
        section1 = workPart.Sections.CreateSection(0.00095, 0.001, 0.050000000000000003)
        
        sweptBuilder1.SectionList.Append(section1)
        
        section1.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.OnlyCurves)
        
        markId44 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId45 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions3 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions3.SetSelectedFromInactive(False)
        
        features1 = [NXOpen.Features.Feature.Null] * 1 
        features1[0] = compositeCurve1
        curveFeatureRule1 = workPart.ScRuleFactory.CreateRuleCurveFeature(features1, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions3)
        
        selectionIntentRuleOptions3.Dispose()
        section1.AllowSelfIntersection(False)
        
        section1.AllowDegenerateCurves(False)
        
        rules3 = [None] * 1 
        rules3[0] = curveFeatureRule1
        helpPoint3 = NXOpen.Point3d(0.0, 0.0, 0.0)
        section1.AddToSection(rules3, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint3, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId45, None)
        
        sections1 = [NXOpen.Section.Null] * 1 
        sections1[0] = section1
        sweptBuilder1.AlignmentMethod.SetSections(sections1)
        
        expression31 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        theSession.DeleteUndoMark(markId44, None)
        
        section1.ReverseDirectionOfLoop(0)
        
        try:
            sweptBuilder1.AlignmentMethod.UpdateSectionAtIndex(0)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1)
            
        section2 = workPart.Sections.CreateSection(0.00095, 0.001, 0.050000000000000003)
        
        sweptBuilder1.GuideList.Append(section2)
        
        section2.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.OnlyCurves)
        
        markId46 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId47 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions4 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions4.SetSelectedFromInactive(False)
        
        features2 = [NXOpen.Features.Feature.Null] * 1 
        features2[0] = compositeCurve2
        curveFeatureRule2 = workPart.ScRuleFactory.CreateRuleCurveFeature(features2, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions4)
        
        selectionIntentRuleOptions4.Dispose()
        section2.AllowSelfIntersection(False)
        
        section2.AllowDegenerateCurves(False)
        
        rules4 = [None] * 1 
        rules4[0] = curveFeatureRule2
        helpPoint4 = NXOpen.Point3d(0.0, 0.0, 0.0)
        section2.AddToSection(rules4, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint4, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId47, None)
        
        sweptBuilder1.ScalingMethod.AreaLaw.AlongSpineData.SetFeatureSpine(section2)
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.AlongSpineData.SetFeatureSpine(section2)
        
        markId48 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId48, "Update Law Data", False)
        
        markId49 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId49, "Update Law Data", False)
        
        sweptBuilder1.OrientationMethod.AngularLaw.AlongSpineData.SetFeatureSpine(section2)
        
        markId50 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId50, "Update Law Data", False)
        
        theSession.DeleteUndoMark(markId46, None)
        
        markId51 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "扫掠")
        
        theSession.DeleteUndoMark(markId51, None)
        
        markId52 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "扫掠")
        
        nXObject8 = sweptBuilder1.Commit()
        
        displayModification1 = theSession.DisplayManager.NewDisplayModification()
        
        displayModification1.ApplyToAllFaces = False
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects9 = [NXOpen.DisplayableObject.Null] * 1 
        swept1 = nXObject8
        face1 = swept1.FindObject("FACE 10001 {(267.9982603648583,3.2867870968743,-1.3918848400633) SWEPT(3)}")
        objects9[0] = face1
        displayModification1.Apply(objects9)
        
        face1.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects10 = [NXOpen.DisplayableObject.Null] * 1 
        face2 = swept1.FindObject("FACE 1 {(0,0.5738425510152,2.4628617099835) SWEPT(3)}")
        objects10[0] = face2
        displayModification1.Apply(objects10)
        
        face2.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects11 = [NXOpen.DisplayableObject.Null] * 1 
        face3 = swept1.FindObject("FACE 10002 {(267.7001687696015,7.1057636173763,-0.8048629507275) SWEPT(3)}")
        objects11[0] = face3
        displayModification1.Apply(objects11)
        
        face3.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects12 = [NXOpen.DisplayableObject.Null] * 1 
        face4 = swept1.FindObject("FACE 2 {(531.6772662032178,47.9072760332953,-19.943130372271) SWEPT(3)}")
        objects12[0] = face4
        displayModification1.Apply(objects12)
        
        face4.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects13 = [NXOpen.DisplayableObject.Null] * 1 
        face5 = swept1.FindObject("FACE 10020 {(268.1428373862207,1.6961176383856,-1.1224915053563) SWEPT(3)}")
        objects13[0] = face5
        displayModification1.Apply(objects13)
        
        face5.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects14 = [NXOpen.DisplayableObject.Null] * 1 
        face6 = swept1.FindObject("FACE 10003 {(267.4029009857479,10.6225125799319,-0.8373437915444) SWEPT(3)}")
        objects14[0] = face6
        displayModification1.Apply(objects14)
        
        face6.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects15 = [NXOpen.DisplayableObject.Null] * 1 
        face7 = swept1.FindObject("FACE 10004 {(267.3104199686494,10.8845983343592,-2.6099282846588) SWEPT(3)}")
        objects15[0] = face7
        displayModification1.Apply(objects15)
        
        face7.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects16 = [NXOpen.DisplayableObject.Null] * 1 
        face8 = swept1.FindObject("FACE 10005 {(267.1246216420692,11.9080142422146,-5.1185485062401) SWEPT(3)}")
        objects16[0] = face8
        displayModification1.Apply(objects16)
        
        face8.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects17 = [NXOpen.DisplayableObject.Null] * 1 
        face9 = swept1.FindObject("FACE 10006 {(267.0604961683621,13.3199414862447,-3.7415893822683) SWEPT(3)}")
        objects17[0] = face9
        displayModification1.Apply(objects17)
        
        face9.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects18 = [NXOpen.DisplayableObject.Null] * 1 
        face10 = swept1.FindObject("FACE 10007 {(267.1413156701853,12.2083335366496,-4.0621567606623) SWEPT(3)}")
        objects18[0] = face10
        displayModification1.Apply(objects18)
        
        face10.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects19 = [NXOpen.DisplayableObject.Null] * 1 
        face11 = swept1.FindObject("FACE 10008 {(267.2183765053112,11.9791054167817,-2.6080992474782) SWEPT(3)}")
        objects19[0] = face11
        displayModification1.Apply(objects19)
        
        face11.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects20 = [NXOpen.DisplayableObject.Null] * 1 
        face12 = swept1.FindObject("FACE 10009 {(267.2662807441716,12.2470909756641,-0.834628949095) SWEPT(3)}")
        objects20[0] = face12
        displayModification1.Apply(objects20)
        
        face12.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects21 = [NXOpen.DisplayableObject.Null] * 1 
        face13 = swept1.FindObject("FACE 10010 {(266.9720500654384,15.7639675750302,-0.7903941755043) SWEPT(3)}")
        objects21[0] = face13
        displayModification1.Apply(objects21)
        
        face13.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects22 = [NXOpen.DisplayableObject.Null] * 1 
        face14 = swept1.FindObject("FACE 10011 {(266.6034592915939,19.6210043692337,-1.8972397949063) SWEPT(3)}")
        objects22[0] = face14
        displayModification1.Apply(objects22)
        
        face14.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects23 = [NXOpen.DisplayableObject.Null] * 1 
        face15 = swept1.FindObject("FACE 10012 {(266.5853462304532,18.853931648111,-3.978118977818) SWEPT(3)}")
        objects23[0] = face15
        displayModification1.Apply(objects23)
        
        face15.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects24 = [NXOpen.DisplayableObject.Null] * 1 
        face16 = swept1.FindObject("FACE 10013 {(266.7470511944819,16.8909701514924,-4.0662709403977) SWEPT(3)}")
        objects24[0] = face16
        displayModification1.Apply(objects24)
        
        face16.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects25 = [NXOpen.DisplayableObject.Null] * 1 
        face17 = swept1.FindObject("FACE 10014 {(266.5133487318046,19.2220748983279,-5.0104525487645) SWEPT(3)}")
        objects25[0] = face17
        displayModification1.Apply(objects25)
        
        face17.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects26 = [NXOpen.DisplayableObject.Null] * 1 
        face18 = swept1.FindObject("FACE 10015 {(266.5452125207065,20.5714792352549,-1.3498512567096) SWEPT(3)}")
        objects26[0] = face18
        displayModification1.Apply(objects26)
        
        face18.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects27 = [NXOpen.DisplayableObject.Null] * 1 
        face19 = swept1.FindObject("FACE 10016 {(267.0054973449585,15.881812483812,0.3011305698042) SWEPT(3)}")
        objects27[0] = face19
        displayModification1.Apply(objects27)
        
        face19.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects28 = [NXOpen.DisplayableObject.Null] * 1 
        face20 = swept1.FindObject("FACE 10017 {(267.3886941334082,11.4370755116175,0.530632602643) SWEPT(3)}")
        objects28[0] = face20
        displayModification1.Apply(objects28)
        
        face20.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects29 = [NXOpen.DisplayableObject.Null] * 1 
        face21 = swept1.FindObject("FACE 10018 {(267.7531311873229,6.9915501463353,0.2862740004665) SWEPT(3)}")
        objects29[0] = face21
        displayModification1.Apply(objects29)
        
        face21.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects30 = [NXOpen.DisplayableObject.Null] * 1 
        face22 = swept1.FindObject("FACE 10019 {(268.0572393541839,3.0958487840862,-0.3118420121523) SWEPT(3)}")
        objects30[0] = face22
        displayModification1.Apply(objects30)
        
        face22.Color = 32767
        
        theSession.DeleteUndoMark(markId52, None)
        
        theSession.SetUndoMarkName(markId43, "扫掠")
        
        sweptBuilder1.Destroy()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression31)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression28)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression29)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression30)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        markId53 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects31 = [NXOpen.DisplayableObject.Null] * 1 
        body1 = workPart.Bodies.FindObject("SWEPT(3)")
        objects31[0] = body1
        theSession.DisplayManager.BlankObjects(objects31)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：文件(F)->导入(M)->文件中的点(L)...
        # ----------------------------------------------
        markId54 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Import Points from File")
        
        pointsFromFileBuilder2 = workPart.CreatePointsFromFileBuilder()
        
        pointsFromFileBuilder2.FileName = prediction_line_path
        
        pointsFromFileBuilder2.CoordinateOption = NXOpen.GeometricUtilities.PointsFromFileBuilder.Options.Absolute
        
        nXObject9 = pointsFromFileBuilder2.Commit()
        
        pointsFromFileBuilder2.Destroy()
        
        nErrs7 = theSession.UpdateManager.DoUpdate(markId54)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->5 拟合曲线
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->曲线(C)->拟合曲线(F)...
        # ----------------------------------------------
        markId55 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        fitCurveBuilder2 = workPart.Features.CreateFitCurveBuilder(NXOpen.Features.FitCurve.Null)
        
        fitCurveBuilder2.Tolerance = 0.001
        
        fitCurveBuilder2.TargetSourceType = NXOpen.Features.FitCurveBuilder.TargetSourceTypes.SpecifiedPoints
        
        fitCurveBuilder2.ProjectionDirectionOption = NXOpen.Features.FitCurveBuilder.ProjectionDirectionOptions.Normal
        
        fitCurveBuilder2.Radius.SetFormula("50")
        
        fitCurveBuilder2.Degree = 24
        
        fitCurveBuilder2.HasUniformSegments = True
        
        fitCurveBuilder2.Extender.StartValue.SetFormula("0")
        
        fitCurveBuilder2.Extender.EndValue.SetFormula("0")
        
        fitCurveBuilder2.RejectionThreshold.SetFormula("10")
        
        fitCurveBuilder2.IsAssociative = False
        
        theSession.SetUndoMarkName(markId55, "拟合曲线 对话框")
        
        markId56 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        theSession.DeleteUndoMark(markId56, None)
        
        markId57 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        # ----------------------------------------------
        #   菜单：编辑(E)->选择(L)->全选(S)
        # ----------------------------------------------
        # Refer to the sample NXOpen application, Selection for "Select All" alternatives.
        objects32 = [NXOpen.TaggedObject.Null] * 302 
        point309 = workPart.Points.FindObject("ENTITY 2 301 1")
        objects32[0] = point309
        point310 = workPart.Points.FindObject("ENTITY 2 300 1")
        objects32[1] = point310
        point311 = workPart.Points.FindObject("ENTITY 2 299 1")
        objects32[2] = point311
        point312 = workPart.Points.FindObject("ENTITY 2 298 1")
        objects32[3] = point312
        point313 = workPart.Points.FindObject("ENTITY 2 297 1")
        objects32[4] = point313
        point314 = workPart.Points.FindObject("ENTITY 2 296 1")
        objects32[5] = point314
        point315 = workPart.Points.FindObject("ENTITY 2 295 1")
        objects32[6] = point315
        point316 = workPart.Points.FindObject("ENTITY 2 294 1")
        objects32[7] = point316
        point317 = workPart.Points.FindObject("ENTITY 2 293 1")
        objects32[8] = point317
        point318 = workPart.Points.FindObject("ENTITY 2 292 1")
        objects32[9] = point318
        point319 = workPart.Points.FindObject("ENTITY 2 291 1")
        objects32[10] = point319
        point320 = workPart.Points.FindObject("ENTITY 2 290 1")
        objects32[11] = point320
        point321 = workPart.Points.FindObject("ENTITY 2 289 1")
        objects32[12] = point321
        point322 = workPart.Points.FindObject("ENTITY 2 288 1")
        objects32[13] = point322
        point323 = workPart.Points.FindObject("ENTITY 2 287 1")
        objects32[14] = point323
        point324 = workPart.Points.FindObject("ENTITY 2 286 1")
        objects32[15] = point324
        point325 = workPart.Points.FindObject("ENTITY 2 285 1")
        objects32[16] = point325
        point326 = workPart.Points.FindObject("ENTITY 2 284 1")
        objects32[17] = point326
        point327 = workPart.Points.FindObject("ENTITY 2 283 1")
        objects32[18] = point327
        point328 = workPart.Points.FindObject("ENTITY 2 282 1")
        objects32[19] = point328
        point329 = workPart.Points.FindObject("ENTITY 2 281 1")
        objects32[20] = point329
        point330 = workPart.Points.FindObject("ENTITY 2 280 1")
        objects32[21] = point330
        point331 = workPart.Points.FindObject("ENTITY 2 279 1")
        objects32[22] = point331
        point332 = workPart.Points.FindObject("ENTITY 2 278 1")
        objects32[23] = point332
        point333 = workPart.Points.FindObject("ENTITY 2 277 1")
        objects32[24] = point333
        point334 = workPart.Points.FindObject("ENTITY 2 276 1")
        objects32[25] = point334
        point335 = workPart.Points.FindObject("ENTITY 2 275 1")
        objects32[26] = point335
        point336 = workPart.Points.FindObject("ENTITY 2 274 1")
        objects32[27] = point336
        point337 = workPart.Points.FindObject("ENTITY 2 273 1")
        objects32[28] = point337
        point338 = workPart.Points.FindObject("ENTITY 2 272 1")
        objects32[29] = point338
        point339 = workPart.Points.FindObject("ENTITY 2 271 1")
        objects32[30] = point339
        point340 = workPart.Points.FindObject("ENTITY 2 270 1")
        objects32[31] = point340
        point341 = workPart.Points.FindObject("ENTITY 2 269 1")
        objects32[32] = point341
        point342 = workPart.Points.FindObject("ENTITY 2 268 1")
        objects32[33] = point342
        point343 = workPart.Points.FindObject("ENTITY 2 267 1")
        objects32[34] = point343
        point344 = workPart.Points.FindObject("ENTITY 2 266 1")
        objects32[35] = point344
        point345 = workPart.Points.FindObject("ENTITY 2 265 1")
        objects32[36] = point345
        point346 = workPart.Points.FindObject("ENTITY 2 264 1")
        objects32[37] = point346
        point347 = workPart.Points.FindObject("ENTITY 2 263 1")
        objects32[38] = point347
        point348 = workPart.Points.FindObject("ENTITY 2 262 1")
        objects32[39] = point348
        point349 = workPart.Points.FindObject("ENTITY 2 261 1")
        objects32[40] = point349
        point350 = workPart.Points.FindObject("ENTITY 2 260 1")
        objects32[41] = point350
        point351 = workPart.Points.FindObject("ENTITY 2 259 1")
        objects32[42] = point351
        point352 = workPart.Points.FindObject("ENTITY 2 258 1")
        objects32[43] = point352
        point353 = workPart.Points.FindObject("ENTITY 2 257 1")
        objects32[44] = point353
        point354 = workPart.Points.FindObject("ENTITY 2 256 1")
        objects32[45] = point354
        point355 = workPart.Points.FindObject("ENTITY 2 255 1")
        objects32[46] = point355
        point356 = workPart.Points.FindObject("ENTITY 2 254 1")
        objects32[47] = point356
        point357 = workPart.Points.FindObject("ENTITY 2 253 1")
        objects32[48] = point357
        point358 = workPart.Points.FindObject("ENTITY 2 252 1")
        objects32[49] = point358
        point359 = workPart.Points.FindObject("ENTITY 2 251 1")
        objects32[50] = point359
        point360 = workPart.Points.FindObject("ENTITY 2 250 1")
        objects32[51] = point360
        point361 = workPart.Points.FindObject("ENTITY 2 249 1")
        objects32[52] = point361
        point362 = workPart.Points.FindObject("ENTITY 2 248 1")
        objects32[53] = point362
        point363 = workPart.Points.FindObject("ENTITY 2 247 1")
        objects32[54] = point363
        point364 = workPart.Points.FindObject("ENTITY 2 246 1")
        objects32[55] = point364
        point365 = workPart.Points.FindObject("ENTITY 2 245 1")
        objects32[56] = point365
        point366 = workPart.Points.FindObject("ENTITY 2 244 1")
        objects32[57] = point366
        point367 = workPart.Points.FindObject("ENTITY 2 243 1")
        objects32[58] = point367
        point368 = workPart.Points.FindObject("ENTITY 2 242 1")
        objects32[59] = point368
        point369 = workPart.Points.FindObject("ENTITY 2 241 1")
        objects32[60] = point369
        point370 = workPart.Points.FindObject("ENTITY 2 240 1")
        objects32[61] = point370
        point371 = workPart.Points.FindObject("ENTITY 2 239 1")
        objects32[62] = point371
        point372 = workPart.Points.FindObject("ENTITY 2 238 1")
        objects32[63] = point372
        point373 = workPart.Points.FindObject("ENTITY 2 237 1")
        objects32[64] = point373
        point374 = workPart.Points.FindObject("ENTITY 2 236 1")
        objects32[65] = point374
        point375 = workPart.Points.FindObject("ENTITY 2 235 1")
        objects32[66] = point375
        point376 = workPart.Points.FindObject("ENTITY 2 234 1")
        objects32[67] = point376
        point377 = workPart.Points.FindObject("ENTITY 2 233 1")
        objects32[68] = point377
        point378 = workPart.Points.FindObject("ENTITY 2 232 1")
        objects32[69] = point378
        point379 = workPart.Points.FindObject("ENTITY 2 231 1")
        objects32[70] = point379
        point380 = workPart.Points.FindObject("ENTITY 2 230 1")
        objects32[71] = point380
        point381 = workPart.Points.FindObject("ENTITY 2 229 1")
        objects32[72] = point381
        point382 = workPart.Points.FindObject("ENTITY 2 228 1")
        objects32[73] = point382
        point383 = workPart.Points.FindObject("ENTITY 2 227 1")
        objects32[74] = point383
        point384 = workPart.Points.FindObject("ENTITY 2 226 1")
        objects32[75] = point384
        point385 = workPart.Points.FindObject("ENTITY 2 225 1")
        objects32[76] = point385
        point386 = workPart.Points.FindObject("ENTITY 2 224 1")
        objects32[77] = point386
        point387 = workPart.Points.FindObject("ENTITY 2 223 1")
        objects32[78] = point387
        point388 = workPart.Points.FindObject("ENTITY 2 222 1")
        objects32[79] = point388
        point389 = workPart.Points.FindObject("ENTITY 2 221 1")
        objects32[80] = point389
        point390 = workPart.Points.FindObject("ENTITY 2 220 1")
        objects32[81] = point390
        point391 = workPart.Points.FindObject("ENTITY 2 219 1")
        objects32[82] = point391
        point392 = workPart.Points.FindObject("ENTITY 2 218 1")
        objects32[83] = point392
        point393 = workPart.Points.FindObject("ENTITY 2 217 1")
        objects32[84] = point393
        point394 = workPart.Points.FindObject("ENTITY 2 216 1")
        objects32[85] = point394
        point395 = workPart.Points.FindObject("ENTITY 2 215 1")
        objects32[86] = point395
        point396 = workPart.Points.FindObject("ENTITY 2 214 1")
        objects32[87] = point396
        point397 = workPart.Points.FindObject("ENTITY 2 213 1")
        objects32[88] = point397
        point398 = workPart.Points.FindObject("ENTITY 2 212 1")
        objects32[89] = point398
        point399 = workPart.Points.FindObject("ENTITY 2 211 1")
        objects32[90] = point399
        point400 = workPart.Points.FindObject("ENTITY 2 210 1")
        objects32[91] = point400
        point401 = workPart.Points.FindObject("ENTITY 2 209 1")
        objects32[92] = point401
        point402 = workPart.Points.FindObject("ENTITY 2 208 1")
        objects32[93] = point402
        point403 = workPart.Points.FindObject("ENTITY 2 207 1")
        objects32[94] = point403
        point404 = workPart.Points.FindObject("ENTITY 2 206 1")
        objects32[95] = point404
        point405 = workPart.Points.FindObject("ENTITY 2 205 1")
        objects32[96] = point405
        point406 = workPart.Points.FindObject("ENTITY 2 204 1")
        objects32[97] = point406
        point407 = workPart.Points.FindObject("ENTITY 2 203 1")
        objects32[98] = point407
        point408 = workPart.Points.FindObject("ENTITY 2 202 1")
        objects32[99] = point408
        point409 = workPart.Points.FindObject("ENTITY 2 201 1")
        objects32[100] = point409
        point410 = workPart.Points.FindObject("ENTITY 2 200 1")
        objects32[101] = point410
        point411 = workPart.Points.FindObject("ENTITY 2 199 1")
        objects32[102] = point411
        point412 = workPart.Points.FindObject("ENTITY 2 198 1")
        objects32[103] = point412
        point413 = workPart.Points.FindObject("ENTITY 2 197 1")
        objects32[104] = point413
        point414 = workPart.Points.FindObject("ENTITY 2 196 1")
        objects32[105] = point414
        point415 = workPart.Points.FindObject("ENTITY 2 195 1")
        objects32[106] = point415
        point416 = workPart.Points.FindObject("ENTITY 2 194 1")
        objects32[107] = point416
        point417 = workPart.Points.FindObject("ENTITY 2 193 1")
        objects32[108] = point417
        point418 = workPart.Points.FindObject("ENTITY 2 192 1")
        objects32[109] = point418
        point419 = workPart.Points.FindObject("ENTITY 2 191 1")
        objects32[110] = point419
        point420 = workPart.Points.FindObject("ENTITY 2 190 1")
        objects32[111] = point420
        point421 = workPart.Points.FindObject("ENTITY 2 189 1")
        objects32[112] = point421
        point422 = workPart.Points.FindObject("ENTITY 2 188 1")
        objects32[113] = point422
        point423 = workPart.Points.FindObject("ENTITY 2 187 1")
        objects32[114] = point423
        point424 = workPart.Points.FindObject("ENTITY 2 186 1")
        objects32[115] = point424
        point425 = workPart.Points.FindObject("ENTITY 2 185 1")
        objects32[116] = point425
        point426 = workPart.Points.FindObject("ENTITY 2 184 1")
        objects32[117] = point426
        point427 = workPart.Points.FindObject("ENTITY 2 183 1")
        objects32[118] = point427
        point428 = workPart.Points.FindObject("ENTITY 2 182 1")
        objects32[119] = point428
        point429 = workPart.Points.FindObject("ENTITY 2 181 1")
        objects32[120] = point429
        point430 = workPart.Points.FindObject("ENTITY 2 180 1")
        objects32[121] = point430
        point431 = workPart.Points.FindObject("ENTITY 2 179 1")
        objects32[122] = point431
        point432 = workPart.Points.FindObject("ENTITY 2 178 1")
        objects32[123] = point432
        point433 = workPart.Points.FindObject("ENTITY 2 177 1")
        objects32[124] = point433
        point434 = workPart.Points.FindObject("ENTITY 2 176 1")
        objects32[125] = point434
        point435 = workPart.Points.FindObject("ENTITY 2 175 1")
        objects32[126] = point435
        point436 = workPart.Points.FindObject("ENTITY 2 174 1")
        objects32[127] = point436
        point437 = workPart.Points.FindObject("ENTITY 2 173 1")
        objects32[128] = point437
        point438 = workPart.Points.FindObject("ENTITY 2 172 1")
        objects32[129] = point438
        point439 = workPart.Points.FindObject("ENTITY 2 171 1")
        objects32[130] = point439
        point440 = workPart.Points.FindObject("ENTITY 2 170 1")
        objects32[131] = point440
        point441 = workPart.Points.FindObject("ENTITY 2 169 1")
        objects32[132] = point441
        point442 = workPart.Points.FindObject("ENTITY 2 168 1")
        objects32[133] = point442
        point443 = workPart.Points.FindObject("ENTITY 2 167 1")
        objects32[134] = point443
        point444 = workPart.Points.FindObject("ENTITY 2 166 1")
        objects32[135] = point444
        point445 = workPart.Points.FindObject("ENTITY 2 165 1")
        objects32[136] = point445
        point446 = workPart.Points.FindObject("ENTITY 2 164 1")
        objects32[137] = point446
        point447 = workPart.Points.FindObject("ENTITY 2 163 1")
        objects32[138] = point447
        point448 = workPart.Points.FindObject("ENTITY 2 162 1")
        objects32[139] = point448
        point449 = workPart.Points.FindObject("ENTITY 2 161 1")
        objects32[140] = point449
        point450 = workPart.Points.FindObject("ENTITY 2 160 1")
        objects32[141] = point450
        point451 = workPart.Points.FindObject("ENTITY 2 159 1")
        objects32[142] = point451
        point452 = workPart.Points.FindObject("ENTITY 2 158 1")
        objects32[143] = point452
        point453 = workPart.Points.FindObject("ENTITY 2 157 1")
        objects32[144] = point453
        point454 = workPart.Points.FindObject("ENTITY 2 156 1")
        objects32[145] = point454
        point455 = workPart.Points.FindObject("ENTITY 2 155 1")
        objects32[146] = point455
        point456 = workPart.Points.FindObject("ENTITY 2 154 1")
        objects32[147] = point456
        point457 = workPart.Points.FindObject("ENTITY 2 153 1")
        objects32[148] = point457
        point458 = workPart.Points.FindObject("ENTITY 2 152 1")
        objects32[149] = point458
        point459 = workPart.Points.FindObject("ENTITY 2 151 1")
        objects32[150] = point459
        point460 = workPart.Points.FindObject("ENTITY 2 150 1")
        objects32[151] = point460
        point461 = workPart.Points.FindObject("ENTITY 2 149 1")
        objects32[152] = point461
        point462 = workPart.Points.FindObject("ENTITY 2 148 1")
        objects32[153] = point462
        point463 = workPart.Points.FindObject("ENTITY 2 147 1")
        objects32[154] = point463
        point464 = workPart.Points.FindObject("ENTITY 2 146 1")
        objects32[155] = point464
        point465 = workPart.Points.FindObject("ENTITY 2 145 1")
        objects32[156] = point465
        point466 = workPart.Points.FindObject("ENTITY 2 144 1")
        objects32[157] = point466
        point467 = workPart.Points.FindObject("ENTITY 2 143 1")
        objects32[158] = point467
        point468 = workPart.Points.FindObject("ENTITY 2 142 1")
        objects32[159] = point468
        point469 = workPart.Points.FindObject("ENTITY 2 141 1")
        objects32[160] = point469
        point470 = workPart.Points.FindObject("ENTITY 2 140 1")
        objects32[161] = point470
        point471 = workPart.Points.FindObject("ENTITY 2 139 1")
        objects32[162] = point471
        point472 = workPart.Points.FindObject("ENTITY 2 138 1")
        objects32[163] = point472
        point473 = workPart.Points.FindObject("ENTITY 2 137 1")
        objects32[164] = point473
        point474 = workPart.Points.FindObject("ENTITY 2 136 1")
        objects32[165] = point474
        point475 = workPart.Points.FindObject("ENTITY 2 135 1")
        objects32[166] = point475
        point476 = workPart.Points.FindObject("ENTITY 2 134 1")
        objects32[167] = point476
        point477 = workPart.Points.FindObject("ENTITY 2 133 1")
        objects32[168] = point477
        point478 = workPart.Points.FindObject("ENTITY 2 132 1")
        objects32[169] = point478
        point479 = workPart.Points.FindObject("ENTITY 2 131 1")
        objects32[170] = point479
        point480 = workPart.Points.FindObject("ENTITY 2 130 1")
        objects32[171] = point480
        point481 = workPart.Points.FindObject("ENTITY 2 129 1")
        objects32[172] = point481
        point482 = workPart.Points.FindObject("ENTITY 2 128 1")
        objects32[173] = point482
        point483 = workPart.Points.FindObject("ENTITY 2 127 1")
        objects32[174] = point483
        point484 = workPart.Points.FindObject("ENTITY 2 126 1")
        objects32[175] = point484
        point485 = workPart.Points.FindObject("ENTITY 2 125 1")
        objects32[176] = point485
        point486 = workPart.Points.FindObject("ENTITY 2 124 1")
        objects32[177] = point486
        point487 = workPart.Points.FindObject("ENTITY 2 123 1")
        objects32[178] = point487
        point488 = workPart.Points.FindObject("ENTITY 2 122 1")
        objects32[179] = point488
        point489 = workPart.Points.FindObject("ENTITY 2 121 1")
        objects32[180] = point489
        point490 = workPart.Points.FindObject("ENTITY 2 120 1")
        objects32[181] = point490
        point491 = workPart.Points.FindObject("ENTITY 2 119 1")
        objects32[182] = point491
        point492 = workPart.Points.FindObject("ENTITY 2 118 1")
        objects32[183] = point492
        point493 = workPart.Points.FindObject("ENTITY 2 117 1")
        objects32[184] = point493
        point494 = workPart.Points.FindObject("ENTITY 2 116 1")
        objects32[185] = point494
        point495 = workPart.Points.FindObject("ENTITY 2 115 1")
        objects32[186] = point495
        point496 = workPart.Points.FindObject("ENTITY 2 114 1")
        objects32[187] = point496
        point497 = workPart.Points.FindObject("ENTITY 2 113 1")
        objects32[188] = point497
        point498 = workPart.Points.FindObject("ENTITY 2 112 1")
        objects32[189] = point498
        point499 = workPart.Points.FindObject("ENTITY 2 111 1")
        objects32[190] = point499
        point500 = workPart.Points.FindObject("ENTITY 2 110 1")
        objects32[191] = point500
        point501 = workPart.Points.FindObject("ENTITY 2 109 1")
        objects32[192] = point501
        point502 = workPart.Points.FindObject("ENTITY 2 108 1")
        objects32[193] = point502
        point503 = workPart.Points.FindObject("ENTITY 2 107 1")
        objects32[194] = point503
        point504 = workPart.Points.FindObject("ENTITY 2 106 1")
        objects32[195] = point504
        point505 = workPart.Points.FindObject("ENTITY 2 105 1")
        objects32[196] = point505
        point506 = workPart.Points.FindObject("ENTITY 2 104 1")
        objects32[197] = point506
        point507 = workPart.Points.FindObject("ENTITY 2 103 1")
        objects32[198] = point507
        point508 = workPart.Points.FindObject("ENTITY 2 102 1")
        objects32[199] = point508
        point509 = workPart.Points.FindObject("ENTITY 2 101 1")
        objects32[200] = point509
        point510 = workPart.Points.FindObject("ENTITY 2 100 1")
        objects32[201] = point510
        point511 = workPart.Points.FindObject("ENTITY 2 99 1")
        objects32[202] = point511
        point512 = workPart.Points.FindObject("ENTITY 2 98 1")
        objects32[203] = point512
        point513 = workPart.Points.FindObject("ENTITY 2 97 1")
        objects32[204] = point513
        point514 = workPart.Points.FindObject("ENTITY 2 96 1")
        objects32[205] = point514
        point515 = workPart.Points.FindObject("ENTITY 2 95 1")
        objects32[206] = point515
        point516 = workPart.Points.FindObject("ENTITY 2 94 1")
        objects32[207] = point516
        point517 = workPart.Points.FindObject("ENTITY 2 93 1")
        objects32[208] = point517
        point518 = workPart.Points.FindObject("ENTITY 2 92 1")
        objects32[209] = point518
        point519 = workPart.Points.FindObject("ENTITY 2 91 1")
        objects32[210] = point519
        point520 = workPart.Points.FindObject("ENTITY 2 90 1")
        objects32[211] = point520
        point521 = workPart.Points.FindObject("ENTITY 2 89 1")
        objects32[212] = point521
        point522 = workPart.Points.FindObject("ENTITY 2 88 1")
        objects32[213] = point522
        point523 = workPart.Points.FindObject("ENTITY 2 87 1")
        objects32[214] = point523
        point524 = workPart.Points.FindObject("ENTITY 2 86 1")
        objects32[215] = point524
        point525 = workPart.Points.FindObject("ENTITY 2 85 1")
        objects32[216] = point525
        point526 = workPart.Points.FindObject("ENTITY 2 84 1")
        objects32[217] = point526
        point527 = workPart.Points.FindObject("ENTITY 2 83 1")
        objects32[218] = point527
        point528 = workPart.Points.FindObject("ENTITY 2 82 1")
        objects32[219] = point528
        point529 = workPart.Points.FindObject("ENTITY 2 81 1")
        objects32[220] = point529
        point530 = workPart.Points.FindObject("ENTITY 2 80 1")
        objects32[221] = point530
        point531 = workPart.Points.FindObject("ENTITY 2 79 1")
        objects32[222] = point531
        point532 = workPart.Points.FindObject("ENTITY 2 78 1")
        objects32[223] = point532
        point533 = workPart.Points.FindObject("ENTITY 2 77 1")
        objects32[224] = point533
        point534 = workPart.Points.FindObject("ENTITY 2 76 1")
        objects32[225] = point534
        point535 = workPart.Points.FindObject("ENTITY 2 75 1")
        objects32[226] = point535
        point536 = workPart.Points.FindObject("ENTITY 2 74 1")
        objects32[227] = point536
        point537 = workPart.Points.FindObject("ENTITY 2 73 1")
        objects32[228] = point537
        point538 = workPart.Points.FindObject("ENTITY 2 72 1")
        objects32[229] = point538
        point539 = workPart.Points.FindObject("ENTITY 2 71 1")
        objects32[230] = point539
        point540 = workPart.Points.FindObject("ENTITY 2 70 1")
        objects32[231] = point540
        point541 = workPart.Points.FindObject("ENTITY 2 69 1")
        objects32[232] = point541
        point542 = workPart.Points.FindObject("ENTITY 2 68 1")
        objects32[233] = point542
        point543 = workPart.Points.FindObject("ENTITY 2 67 1")
        objects32[234] = point543
        point544 = workPart.Points.FindObject("ENTITY 2 66 1")
        objects32[235] = point544
        point545 = workPart.Points.FindObject("ENTITY 2 65 1")
        objects32[236] = point545
        point546 = workPart.Points.FindObject("ENTITY 2 64 1")
        objects32[237] = point546
        point547 = workPart.Points.FindObject("ENTITY 2 63 1")
        objects32[238] = point547
        point548 = workPart.Points.FindObject("ENTITY 2 62 1")
        objects32[239] = point548
        point549 = workPart.Points.FindObject("ENTITY 2 61 1")
        objects32[240] = point549
        point550 = workPart.Points.FindObject("ENTITY 2 60 1")
        objects32[241] = point550
        point551 = workPart.Points.FindObject("ENTITY 2 59 1")
        objects32[242] = point551
        point552 = workPart.Points.FindObject("ENTITY 2 58 1")
        objects32[243] = point552
        point553 = workPart.Points.FindObject("ENTITY 2 57 1")
        objects32[244] = point553
        point554 = workPart.Points.FindObject("ENTITY 2 56 1")
        objects32[245] = point554
        point555 = workPart.Points.FindObject("ENTITY 2 55 1")
        objects32[246] = point555
        point556 = workPart.Points.FindObject("ENTITY 2 54 1")
        objects32[247] = point556
        point557 = workPart.Points.FindObject("ENTITY 2 53 1")
        objects32[248] = point557
        point558 = workPart.Points.FindObject("ENTITY 2 52 1")
        objects32[249] = point558
        point559 = workPart.Points.FindObject("ENTITY 2 51 1")
        objects32[250] = point559
        point560 = workPart.Points.FindObject("ENTITY 2 50 1")
        objects32[251] = point560
        point561 = workPart.Points.FindObject("ENTITY 2 49 1")
        objects32[252] = point561
        point562 = workPart.Points.FindObject("ENTITY 2 48 1")
        objects32[253] = point562
        point563 = workPart.Points.FindObject("ENTITY 2 47 1")
        objects32[254] = point563
        point564 = workPart.Points.FindObject("ENTITY 2 46 1")
        objects32[255] = point564
        point565 = workPart.Points.FindObject("ENTITY 2 45 1")
        objects32[256] = point565
        point566 = workPart.Points.FindObject("ENTITY 2 44 1")
        objects32[257] = point566
        point567 = workPart.Points.FindObject("ENTITY 2 43 1")
        objects32[258] = point567
        point568 = workPart.Points.FindObject("ENTITY 2 42 1")
        objects32[259] = point568
        point569 = workPart.Points.FindObject("ENTITY 2 41 1")
        objects32[260] = point569
        point570 = workPart.Points.FindObject("ENTITY 2 40 1")
        objects32[261] = point570
        point571 = workPart.Points.FindObject("ENTITY 2 39 1")
        objects32[262] = point571
        point572 = workPart.Points.FindObject("ENTITY 2 38 1")
        objects32[263] = point572
        point573 = workPart.Points.FindObject("ENTITY 2 37 1")
        objects32[264] = point573
        point574 = workPart.Points.FindObject("ENTITY 2 36 1")
        objects32[265] = point574
        point575 = workPart.Points.FindObject("ENTITY 2 35 1")
        objects32[266] = point575
        point576 = workPart.Points.FindObject("ENTITY 2 34 1")
        objects32[267] = point576
        point577 = workPart.Points.FindObject("ENTITY 2 33 1")
        objects32[268] = point577
        point578 = workPart.Points.FindObject("ENTITY 2 32 1")
        objects32[269] = point578
        point579 = workPart.Points.FindObject("ENTITY 2 31 1")
        objects32[270] = point579
        point580 = workPart.Points.FindObject("ENTITY 2 30 1")
        objects32[271] = point580
        point581 = workPart.Points.FindObject("ENTITY 2 29 1")
        objects32[272] = point581
        point582 = workPart.Points.FindObject("ENTITY 2 28 1")
        objects32[273] = point582
        point583 = workPart.Points.FindObject("ENTITY 2 27 1")
        objects32[274] = point583
        point584 = workPart.Points.FindObject("ENTITY 2 26 1")
        objects32[275] = point584
        point585 = workPart.Points.FindObject("ENTITY 2 25 1")
        objects32[276] = point585
        point586 = workPart.Points.FindObject("ENTITY 2 24 1")
        objects32[277] = point586
        point587 = workPart.Points.FindObject("ENTITY 2 23 1")
        objects32[278] = point587
        point588 = workPart.Points.FindObject("ENTITY 2 22 1")
        objects32[279] = point588
        point589 = workPart.Points.FindObject("ENTITY 2 21 1")
        objects32[280] = point589
        point590 = workPart.Points.FindObject("ENTITY 2 20 1")
        objects32[281] = point590
        point591 = workPart.Points.FindObject("ENTITY 2 19 1")
        objects32[282] = point591
        point592 = workPart.Points.FindObject("ENTITY 2 18 1")
        objects32[283] = point592
        point593 = workPart.Points.FindObject("ENTITY 2 17 1")
        objects32[284] = point593
        point594 = workPart.Points.FindObject("ENTITY 2 16 1")
        objects32[285] = point594
        point595 = workPart.Points.FindObject("ENTITY 2 15 1")
        objects32[286] = point595
        point596 = workPart.Points.FindObject("ENTITY 2 14 1")
        objects32[287] = point596
        point597 = workPart.Points.FindObject("ENTITY 2 13 1")
        objects32[288] = point597
        point598 = workPart.Points.FindObject("ENTITY 2 12 1")
        objects32[289] = point598
        point599 = workPart.Points.FindObject("ENTITY 2 11 1")
        objects32[290] = point599
        point600 = workPart.Points.FindObject("ENTITY 2 10 1")
        objects32[291] = point600
        point601 = workPart.Points.FindObject("ENTITY 2 9 1")
        objects32[292] = point601
        point602 = workPart.Points.FindObject("ENTITY 2 8 1")
        objects32[293] = point602
        point603 = workPart.Points.FindObject("ENTITY 2 7 1")
        objects32[294] = point603
        point604 = workPart.Points.FindObject("ENTITY 2 6 1")
        objects32[295] = point604
        point605 = workPart.Points.FindObject("ENTITY 2 5 1")
        objects32[296] = point605
        point606 = workPart.Points.FindObject("ENTITY 2 4 1")
        objects32[297] = point606
        point607 = workPart.Points.FindObject("ENTITY 2 3 1")
        objects32[298] = point607
        point608 = workPart.Points.FindObject("ENTITY 2 2 1")
        objects32[299] = point608
        point609 = workPart.Points.FindObject("ENTITY 2 1 1")
        objects32[300] = point609
        group2 = nXObject9
        objects32[301] = group2
        added3 = fitCurveBuilder2.Target.Add(objects32)
        
        geometricConstraintData3 = fitCurveBuilder2.ConstraintManager.FindItem(0)
        
        point610 = geometricConstraintData3.Point
        
        geometricConstraintData4 = fitCurveBuilder2.ConstraintManager.FindItem(1)
        
        point611 = geometricConstraintData4.Point
        
        theSession.SetUndoMarkName(markId57, "拟合曲线 - 选择")
        
        theSession.SetUndoMarkVisibility(markId57, None, NXOpen.Session.MarkVisibility.Visible)
        
        theSession.SetUndoMarkVisibility(markId55, None, NXOpen.Session.MarkVisibility.Invisible)
        
        markId58 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        theSession.DeleteUndoMark(markId58, None)
        
        markId59 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        fitCurveBuilder2.HasReversedDirection = True
        
        theSession.SetUndoMarkName(markId59, "拟合曲线 - 反向")
        
        theSession.SetUndoMarkVisibility(markId59, None, NXOpen.Session.MarkVisibility.Visible)
        
        theSession.SetUndoMarkVisibility(markId55, None, NXOpen.Session.MarkVisibility.Invisible)
        
        markId60 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        markId61 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        theSession.DeleteUndoMark(markId61, None)
        
        markId62 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        nXObject10 = fitCurveBuilder2.Commit()
        
        theSession.DeleteUndoMark(markId62, None)
        
        theSession.SetUndoMarkName(markId55, "拟合曲线")
        
        fitCurveBuilder2.Destroy()
        
        theSession.SetUndoMarkVisibility(markId55, None, NXOpen.Session.MarkVisibility.Visible)
        
        theSession.DeleteUndoMark(markId59, None)
        
        theSession.DeleteUndoMark(markId57, None)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->5 抽取几何特征
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->关联复制(A)->抽取几何特征(E)...
        # ----------------------------------------------
        markId63 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        wavePointBuilder3 = workPart.Features.CreateWavePointBuilder(NXOpen.Features.Feature.Null)
        
        waveDatumBuilder3 = workPart.Features.CreateWaveDatumBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder3 = workPart.Features.CreateCompositeCurveBuilder(NXOpen.Features.Feature.Null)
        
        extractFaceBuilder3 = workPart.Features.CreateExtractFaceBuilder(NXOpen.Features.Feature.Null)
        
        mirrorBodyBuilder3 = workPart.Features.CreateMirrorBodyBuilder(NXOpen.Features.Feature.Null)
        
        waveSketchBuilder3 = workPart.Features.CreateWaveSketchBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder3.CurveFitData.Tolerance = 0.001
        
        compositeCurveBuilder3.CurveFitData.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder3.Section.SetAllowRefCrvs(False)
        
        extractFaceBuilder3.FaceOption = NXOpen.Features.ExtractFaceBuilder.FaceOptionType.AdjacentFaces
        
        compositeCurveBuilder3.Associative = False
        
        waveDatumBuilder3.ParentPart = NXOpen.Features.WaveDatumBuilder.ParentPartType.WorkPart
        
        wavePointBuilder3.ParentPart = NXOpen.Features.WavePointBuilder.ParentPartType.WorkPart
        
        extractFaceBuilder3.ParentPart = NXOpen.Features.ExtractFaceBuilder.ParentPartType.WorkPart
        
        mirrorBodyBuilder3.ParentPartType = NXOpen.Features.MirrorBodyBuilder.ParentPart.WorkPart
        
        compositeCurveBuilder3.ParentPart = NXOpen.Features.CompositeCurveBuilder.PartType.WorkPart
        
        waveSketchBuilder3.ParentPart = NXOpen.Features.WaveSketchBuilder.ParentPartType.WorkPart
        
        compositeCurveBuilder3.Associative = False
        
        theSession.SetUndoMarkName(markId63, "抽取几何特征 对话框")
        
        compositeCurveBuilder3.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder3.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder3.Section.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder3.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder3.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder3.Associative = False
        
        compositeCurveBuilder3.FixAtCurrentTimestamp = False
        
        compositeCurveBuilder3.HideOriginal = False
        
        compositeCurveBuilder3.InheritDisplayProperties = False
        
        compositeCurveBuilder3.Section.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.CurvesAndPoints)
        
        markId64 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId65 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions5 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions5.SetSelectedFromInactive(False)
        
        curves3 = [NXOpen.IBaseCurve.Null] * 1 
        spline3 = workPart.Splines.FindObject("ENTITY 9 1 1")
        curves3[0] = spline3
        curveDumbRule3 = workPart.ScRuleFactory.CreateRuleBaseCurveDumb(curves3, selectionIntentRuleOptions5)
        
        selectionIntentRuleOptions5.Dispose()
        compositeCurveBuilder3.Section.AllowSelfIntersection(False)
        
        compositeCurveBuilder3.Section.AllowDegenerateCurves(False)
        
        rules5 = [None] * 1 
        rules5[0] = curveDumbRule3
        helpPoint5 = NXOpen.Point3d(43.662575329853254, 0.33453176538116813, -0.15875637291874128)
        compositeCurveBuilder3.Section.AddToSection(rules5, spline3, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint5, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId65, None)
        
        theSession.DeleteUndoMark(markId64, None)
        
        markId66 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        theSession.DeleteUndoMark(markId66, None)
        
        markId67 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        nXObject11 = compositeCurveBuilder3.Commit()
        
        theSession.DeleteUndoMark(markId67, None)
        
        theSession.SetUndoMarkName(markId63, "抽取几何特征")
        
        compositeCurveBuilder3.Destroy()
        
        waveDatumBuilder3.Destroy()
        
        wavePointBuilder3.Destroy()
        
        extractFaceBuilder3.Destroy()
        
        mirrorBodyBuilder3.Destroy()
        
        waveSketchBuilder3.Destroy()
        
        markId68 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects33 = [NXOpen.DisplayableObject.Null] * 1 
        compositeCurve3 = nXObject11
        spline4 = compositeCurve3.FindObject("CURVE 1")
        objects33[0] = spline4
        theSession.DisplayManager.BlankObjects(objects33)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：编辑(E)->选择(L)->全选(S)
        # ----------------------------------------------
        # Refer to the sample NXOpen application, Selection for "Select All" alternatives.
        # ----------------------------------------------
        #   菜单：编辑(E)->删除(D)...
        # ----------------------------------------------
        markId69 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Delete")
        
        theSession.UpdateManager.ClearErrorList()
        
        markId70 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Delete")
        
        objects34 = [NXOpen.TaggedObject.Null] * 303 
        objects34[0] = point554
        objects34[1] = point555
        objects34[2] = point556
        objects34[3] = point557
        objects34[4] = point558
        objects34[5] = point559
        objects34[6] = point560
        objects34[7] = point561
        objects34[8] = point562
        objects34[9] = point563
        objects34[10] = point564
        objects34[11] = point565
        objects34[12] = point566
        objects34[13] = point567
        objects34[14] = point568
        objects34[15] = point569
        objects34[16] = point570
        objects34[17] = point571
        objects34[18] = point572
        objects34[19] = point573
        objects34[20] = point574
        objects34[21] = point575
        objects34[22] = point576
        objects34[23] = point577
        objects34[24] = point578
        objects34[25] = point579
        objects34[26] = point580
        objects34[27] = point581
        objects34[28] = point582
        objects34[29] = point583
        objects34[30] = point584
        objects34[31] = point585
        objects34[32] = point586
        objects34[33] = point587
        objects34[34] = point588
        objects34[35] = point589
        objects34[36] = point590
        objects34[37] = point591
        objects34[38] = point592
        objects34[39] = point593
        objects34[40] = point594
        objects34[41] = point595
        objects34[42] = point596
        objects34[43] = point597
        objects34[44] = point598
        objects34[45] = point599
        objects34[46] = point600
        objects34[47] = point601
        objects34[48] = point602
        objects34[49] = point603
        objects34[50] = point604
        objects34[51] = point605
        objects34[52] = point606
        objects34[53] = point607
        objects34[54] = point608
        objects34[55] = point609
        objects34[56] = group2
        objects34[57] = point475
        objects34[58] = point476
        objects34[59] = point477
        objects34[60] = point478
        objects34[61] = point479
        objects34[62] = point480
        objects34[63] = point481
        objects34[64] = point482
        objects34[65] = point483
        objects34[66] = point484
        objects34[67] = point485
        objects34[68] = point486
        objects34[69] = point487
        objects34[70] = point488
        objects34[71] = point489
        objects34[72] = point490
        objects34[73] = point491
        objects34[74] = point492
        objects34[75] = point493
        objects34[76] = point494
        objects34[77] = point495
        objects34[78] = point496
        objects34[79] = point497
        objects34[80] = point498
        objects34[81] = point499
        objects34[82] = point500
        objects34[83] = point501
        objects34[84] = point502
        objects34[85] = point503
        objects34[86] = point504
        objects34[87] = point505
        objects34[88] = point506
        objects34[89] = point507
        objects34[90] = point508
        objects34[91] = point509
        objects34[92] = point510
        objects34[93] = point511
        objects34[94] = point512
        objects34[95] = point513
        objects34[96] = point514
        objects34[97] = point515
        objects34[98] = point516
        objects34[99] = point517
        objects34[100] = point518
        objects34[101] = point519
        objects34[102] = point384
        objects34[103] = point385
        objects34[104] = point386
        objects34[105] = point387
        objects34[106] = point388
        objects34[107] = point389
        objects34[108] = point390
        objects34[109] = point391
        objects34[110] = point392
        objects34[111] = point393
        objects34[112] = point394
        objects34[113] = point395
        objects34[114] = point396
        objects34[115] = point397
        objects34[116] = point398
        objects34[117] = point399
        objects34[118] = point400
        objects34[119] = point401
        objects34[120] = point402
        objects34[121] = point403
        objects34[122] = point404
        objects34[123] = point405
        objects34[124] = point406
        objects34[125] = point407
        objects34[126] = point408
        objects34[127] = point409
        objects34[128] = point410
        objects34[129] = point411
        objects34[130] = point412
        objects34[131] = point413
        objects34[132] = point414
        objects34[133] = point415
        objects34[134] = point416
        objects34[135] = point417
        objects34[136] = point418
        objects34[137] = point419
        objects34[138] = point420
        objects34[139] = point421
        objects34[140] = point422
        objects34[141] = point423
        objects34[142] = point424
        objects34[143] = point425
        objects34[144] = point426
        objects34[145] = point427
        objects34[146] = point428
        objects34[147] = point429
        objects34[148] = point430
        objects34[149] = point431
        objects34[150] = point432
        objects34[151] = point433
        objects34[152] = point434
        objects34[153] = point435
        objects34[154] = point436
        objects34[155] = point437
        objects34[156] = point438
        objects34[157] = point357
        objects34[158] = point358
        objects34[159] = point359
        objects34[160] = point360
        objects34[161] = point361
        objects34[162] = point362
        objects34[163] = point363
        objects34[164] = point364
        objects34[165] = point365
        objects34[166] = point366
        objects34[167] = point309
        objects34[168] = point310
        objects34[169] = point311
        objects34[170] = point312
        objects34[171] = point313
        objects34[172] = point314
        objects34[173] = point315
        objects34[174] = point316
        objects34[175] = point317
        objects34[176] = point318
        objects34[177] = point319
        objects34[178] = point320
        objects34[179] = point321
        objects34[180] = point322
        objects34[181] = point323
        objects34[182] = point324
        objects34[183] = point325
        objects34[184] = point326
        objects34[185] = point327
        objects34[186] = point328
        objects34[187] = point329
        objects34[188] = point330
        objects34[189] = point331
        objects34[190] = point332
        objects34[191] = point333
        objects34[192] = point334
        objects34[193] = point335
        objects34[194] = point336
        objects34[195] = point337
        objects34[196] = point338
        objects34[197] = point339
        objects34[198] = point340
        objects34[199] = point341
        objects34[200] = point342
        objects34[201] = point343
        objects34[202] = point344
        objects34[203] = point345
        objects34[204] = point346
        objects34[205] = point347
        objects34[206] = point348
        objects34[207] = point349
        objects34[208] = point350
        objects34[209] = point351
        objects34[210] = point352
        objects34[211] = point353
        objects34[212] = point354
        objects34[213] = point355
        objects34[214] = point356
        objects34[215] = spline3
        objects34[216] = point520
        objects34[217] = point521
        objects34[218] = point522
        objects34[219] = point523
        objects34[220] = point524
        objects34[221] = point525
        objects34[222] = point526
        objects34[223] = point527
        objects34[224] = point528
        objects34[225] = point529
        objects34[226] = point530
        objects34[227] = point531
        objects34[228] = point532
        objects34[229] = point533
        objects34[230] = point534
        objects34[231] = point535
        objects34[232] = point536
        objects34[233] = point537
        objects34[234] = point538
        objects34[235] = point539
        objects34[236] = point540
        objects34[237] = point541
        objects34[238] = point542
        objects34[239] = point543
        objects34[240] = point544
        objects34[241] = point545
        objects34[242] = point546
        objects34[243] = point547
        objects34[244] = point548
        objects34[245] = point549
        objects34[246] = point550
        objects34[247] = point551
        objects34[248] = point552
        objects34[249] = point553
        objects34[250] = point439
        objects34[251] = point440
        objects34[252] = point441
        objects34[253] = point442
        objects34[254] = point443
        objects34[255] = point444
        objects34[256] = point445
        objects34[257] = point446
        objects34[258] = point447
        objects34[259] = point448
        objects34[260] = point449
        objects34[261] = point450
        objects34[262] = point451
        objects34[263] = point452
        objects34[264] = point453
        objects34[265] = point454
        objects34[266] = point455
        objects34[267] = point456
        objects34[268] = point457
        objects34[269] = point458
        objects34[270] = point459
        objects34[271] = point460
        objects34[272] = point461
        objects34[273] = point462
        objects34[274] = point463
        objects34[275] = point464
        objects34[276] = point465
        objects34[277] = point466
        objects34[278] = point467
        objects34[279] = point468
        objects34[280] = point469
        objects34[281] = point470
        objects34[282] = point471
        objects34[283] = point472
        objects34[284] = point473
        objects34[285] = point474
        objects34[286] = point367
        objects34[287] = point368
        objects34[288] = point369
        objects34[289] = point370
        objects34[290] = point371
        objects34[291] = point372
        objects34[292] = point373
        objects34[293] = point374
        objects34[294] = point375
        objects34[295] = point376
        objects34[296] = point377
        objects34[297] = point378
        objects34[298] = point379
        objects34[299] = point380
        objects34[300] = point381
        objects34[301] = point382
        objects34[302] = point383
        nErrs8 = theSession.UpdateManager.AddObjectsToDeleteList(objects34)
        
        id3 = theSession.NewestVisibleUndoMark
        
        nErrs9 = theSession.UpdateManager.DoUpdate(id3)
        
        theSession.DeleteUndoMark(markId69, None)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->5 扫掠
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->扫掠(W)->扫掠(S)...
        # ----------------------------------------------
        markId71 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        sweptBuilder2 = workPart.Features.CreateSweptBuilder(NXOpen.Features.Swept.Null)
        
        expression32 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        expression33 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        expression34 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        sweptBuilder2.G0Tolerance = 0.001
        
        sweptBuilder2.G1Tolerance = 0.050000000000000003
        
        sweptBuilder2.OrientationMethod.AngularLaw.Value.SetFormula("0")
        
        sweptBuilder2.OrientationMethod.AngularLaw.StartValue.SetFormula("0")
        
        sweptBuilder2.OrientationMethod.AngularLaw.EndValue.SetFormula("0")
        
        sweptBuilder2.ScalingMethod.AreaLaw.Value.SetFormula("1")
        
        sweptBuilder2.ScalingMethod.AreaLaw.StartValue.SetFormula("1")
        
        sweptBuilder2.ScalingMethod.AreaLaw.EndValue.SetFormula("1")
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.Value.SetFormula("1")
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.StartValue.SetFormula("1")
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.EndValue.SetFormula("1")
        
        theSession.SetUndoMarkName(markId71, "扫掠 对话框")
        
        sweptBuilder2.Spine.DistanceTolerance = 0.001
        
        sweptBuilder2.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder2.AlignmentMethod.AlignCurve.DistanceTolerance = 0.001
        
        sweptBuilder2.AlignmentMethod.AlignCurve.ChainingTolerance = 0.00095
        
        sweptBuilder2.OrientationMethod.OrientationCurve.DistanceTolerance = 0.001
        
        sweptBuilder2.OrientationMethod.OrientationCurve.ChainingTolerance = 0.00095
        
        sweptBuilder2.OrientationMethod.AngularLaw.AlongSpineData.Spine.DistanceTolerance = 0.001
        
        sweptBuilder2.OrientationMethod.AngularLaw.AlongSpineData.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder2.OrientationMethod.AngularLaw.LawCurve.DistanceTolerance = 0.001
        
        sweptBuilder2.OrientationMethod.AngularLaw.LawCurve.ChainingTolerance = 0.00095
        
        sweptBuilder2.ScalingMethod.ScalingCurve.DistanceTolerance = 0.001
        
        sweptBuilder2.ScalingMethod.ScalingCurve.ChainingTolerance = 0.00095
        
        sweptBuilder2.ScalingMethod.AreaLaw.AlongSpineData.Spine.DistanceTolerance = 0.001
        
        sweptBuilder2.ScalingMethod.AreaLaw.AlongSpineData.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder2.ScalingMethod.AreaLaw.LawCurve.DistanceTolerance = 0.001
        
        sweptBuilder2.ScalingMethod.AreaLaw.LawCurve.ChainingTolerance = 0.00095
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.AlongSpineData.Spine.DistanceTolerance = 0.001
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.AlongSpineData.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.LawCurve.DistanceTolerance = 0.001
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.LawCurve.ChainingTolerance = 0.00095
        
        sweptBuilder2.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.AlignmentMethod.AlignCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.OrientationMethod.OrientationCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.OrientationMethod.AngularLaw.AlongSpineData.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.OrientationMethod.AngularLaw.LawCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.ScalingMethod.ScalingCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.ScalingMethod.AreaLaw.AlongSpineData.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.ScalingMethod.AreaLaw.LawCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.AlongSpineData.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.LawCurve.AngleTolerance = 0.050000000000000003
        
        section3 = workPart.Sections.CreateSection(0.00095, 0.001, 0.050000000000000003)
        
        sweptBuilder2.SectionList.Append(section3)
        
        section3.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.OnlyCurves)
        
        markId72 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId73 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions6 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions6.SetSelectedFromInactive(False)
        
        features3 = [NXOpen.Features.Feature.Null] * 1 
        features3[0] = compositeCurve1
        curveFeatureRule3 = workPart.ScRuleFactory.CreateRuleCurveFeature(features3, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions6)
        
        selectionIntentRuleOptions6.Dispose()
        section3.AllowSelfIntersection(False)
        
        section3.AllowDegenerateCurves(False)
        
        rules6 = [None] * 1 
        rules6[0] = curveFeatureRule3
        helpPoint6 = NXOpen.Point3d(0.0, 0.0, 0.0)
        section3.AddToSection(rules6, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint6, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId73, None)
        
        sections2 = [NXOpen.Section.Null] * 1 
        sections2[0] = section3
        sweptBuilder2.AlignmentMethod.SetSections(sections2)
        
        expression35 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        theSession.DeleteUndoMark(markId72, None)
        
        section3.ReverseDirectionOfLoop(0)
        
        try:
            sweptBuilder2.AlignmentMethod.UpdateSectionAtIndex(0)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1)
            
        section4 = workPart.Sections.CreateSection(0.00095, 0.001, 0.050000000000000003)
        
        sweptBuilder2.GuideList.Append(section4)
        
        section4.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.OnlyCurves)
        
        markId74 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId75 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions7 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions7.SetSelectedFromInactive(False)
        
        features4 = [NXOpen.Features.Feature.Null] * 1 
        features4[0] = compositeCurve3
        curveFeatureRule4 = workPart.ScRuleFactory.CreateRuleCurveFeature(features4, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions7)
        
        selectionIntentRuleOptions7.Dispose()
        section4.AllowSelfIntersection(False)
        
        section4.AllowDegenerateCurves(False)
        
        rules7 = [None] * 1 
        rules7[0] = curveFeatureRule4
        helpPoint7 = NXOpen.Point3d(0.0, 0.0, 0.0)
        section4.AddToSection(rules7, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint7, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId75, None)
        
        sweptBuilder2.ScalingMethod.AreaLaw.AlongSpineData.SetFeatureSpine(section4)
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.AlongSpineData.SetFeatureSpine(section4)
        
        markId76 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId76, "Update Law Data", False)
        
        markId77 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId77, "Update Law Data", False)
        
        sweptBuilder2.OrientationMethod.AngularLaw.AlongSpineData.SetFeatureSpine(section4)
        
        markId78 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId78, "Update Law Data", False)
        
        theSession.DeleteUndoMark(markId74, None)
        
        markId79 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "扫掠")
        
        theSession.DeleteUndoMark(markId79, None)
        
        markId80 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "扫掠")
        
        nXObject12 = sweptBuilder2.Commit()
        
        displayModification2 = theSession.DisplayManager.NewDisplayModification()
        
        displayModification2.ApplyToAllFaces = False
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects35 = [NXOpen.DisplayableObject.Null] * 1 
        swept2 = nXObject12
        face23 = swept2.FindObject("FACE 10001 {(267.9387934167045,3.2325000449176,-1.2846236378605) SWEPT(5)}")
        objects35[0] = face23
        displayModification2.Apply(objects35)
        
        face23.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects36 = [NXOpen.DisplayableObject.Null] * 1 
        face24 = swept2.FindObject("FACE 1 {(0,0.5738425510152,2.4628617099835) SWEPT(5)}")
        objects36[0] = face24
        displayModification2.Apply(objects36)
        
        face24.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects37 = [NXOpen.DisplayableObject.Null] * 1 
        face25 = swept2.FindObject("FACE 10002 {(267.6419689376582,7.0515968767719,-0.6977411727779) SWEPT(5)}")
        objects37[0] = face25
        displayModification2.Apply(objects37)
        
        face25.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects38 = [NXOpen.DisplayableObject.Null] * 1 
        face26 = swept2.FindObject("FACE 2 {(531.6483905994322,47.6007792805876,-19.2465504913904) SWEPT(5)}")
        objects38[0] = face26
        displayModification2.Apply(objects38)
        
        face26.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects39 = [NXOpen.DisplayableObject.Null] * 1 
        face27 = swept2.FindObject("FACE 10020 {(268.0826231480158,1.6417740150861,-1.0151634372384) SWEPT(5)}")
        objects39[0] = face27
        displayModification2.Apply(objects39)
        
        face27.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects40 = [NXOpen.DisplayableObject.Null] * 1 
        face28 = swept2.FindObject("FACE 10003 {(267.3461127725446,10.5684638325171,-0.7303602094485) SWEPT(5)}")
        objects40[0] = face28
        displayModification2.Apply(objects40)
        
        face28.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects41 = [NXOpen.DisplayableObject.Null] * 1 
        face29 = swept2.FindObject("FACE 10004 {(267.2544930914269,10.8305806309864,-2.502985290898) SWEPT(5)}")
        objects41[0] = face29
        displayModification2.Apply(objects41)
        
        face29.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects42 = [NXOpen.DisplayableObject.Null] * 1 
        face30 = swept2.FindObject("FACE 10005 {(267.0701730958501,11.854062290298,-5.0116884924034) SWEPT(5)}")
        objects42[0] = face30
        displayModification2.Apply(objects42)
        
        face30.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects43 = [NXOpen.DisplayableObject.Null] * 1 
        face31 = swept2.FindObject("FACE 10006 {(267.0060206220967,13.2660194349884,-3.6347610679052) SWEPT(5)}")
        objects43[0] = face31
        displayModification2.Apply(objects43)
        
        face31.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects44 = [NXOpen.DisplayableObject.Null] * 1 
        face32 = swept2.FindObject("FACE 10007 {(267.0865352434817,12.154378347486,-3.9552904249273) SWEPT(5)}")
        objects44[0] = face32
        displayModification2.Apply(objects44)
        
        face32.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects45 = [NXOpen.DisplayableObject.Null] * 1 
        face33 = swept2.FindObject("FACE 10008 {(267.1628838625351,11.9251242860128,-2.5011990597222) SWEPT(5)}")
        objects45[0] = face33
        displayModification2.Apply(objects45)
        
        face33.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects46 = [NXOpen.DisplayableObject.Null] * 1 
        face34 = swept2.FindObject("FACE 10009 {(267.2101370657654,12.1930965130165,-0.7277089040119) SWEPT(5)}")
        objects46[0] = face34
        displayModification2.Apply(objects46)
        
        face34.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects47 = [NXOpen.DisplayableObject.Null] * 1 
        face35 = swept2.FindObject("FACE 10010 {(266.9172852870117,15.7100901455402,-0.6836110185889) SWEPT(5)}")
        objects47[0] = face35
        displayModification2.Apply(objects47)
        
        face35.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects48 = [NXOpen.DisplayableObject.Null] * 1 
        face36 = swept2.FindObject("FACE 10011 {(266.550700301249,19.5672698152361,-1.7906265356442) SWEPT(5)}")
        objects48[0] = face36
        displayModification2.Apply(objects48)
        
        face36.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects49 = [NXOpen.DisplayableObject.Null] * 1 
        face37 = swept2.FindObject("FACE 10012 {(266.5331712231418,18.8001976027375,-3.8715113024652) SWEPT(5)}")
        objects49[0] = face37
        displayModification2.Apply(objects49)
        
        face37.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects50 = [NXOpen.DisplayableObject.Null] * 1 
        face38 = swept2.FindObject("FACE 10013 {(266.694133655736,16.837171581208,-3.9595879461599) SWEPT(5)}")
        objects50[0] = face38
        displayModification2.Apply(objects50)
        
        face38.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects51 = [NXOpen.DisplayableObject.Null] * 1 
        face39 = swept2.FindObject("FACE 10014 {(266.4617610112121,19.1683661382986,-4.9038769462326) SWEPT(5)}")
        objects51[0] = face39
        displayModification2.Apply(objects51)
        
        face39.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects52 = [NXOpen.DisplayableObject.Null] * 1 
        face40 = swept2.FindObject("FACE 10015 {(266.4925974799962,20.5177695804737,-1.243265831183) SWEPT(5)}")
        objects52[0] = face40
        displayModification2.Apply(objects52)
        
        face40.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects53 = [NXOpen.DisplayableObject.Null] * 1 
        face41 = swept2.FindObject("FACE 10016 {(266.9503131537033,15.8279252743041,0.407927791542) SWEPT(5)}")
        objects53[0] = face41
        displayModification2.Apply(objects53)
        
        face41.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects54 = [NXOpen.DisplayableObject.Null] * 1 
        face42 = swept2.FindObject("FACE 10017 {(267.3316453326115,11.3830368044887,0.6376077114701) SWEPT(5)}")
        objects54[0] = face42
        displayModification2.Apply(objects54)
        
        face42.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects55 = [NXOpen.DisplayableObject.Null] * 1 
        face43 = swept2.FindObject("FACE 10018 {(267.6944198757715,6.9373658715567,0.3934189190063) SWEPT(5)}")
        objects55[0] = face43
        displayModification2.Apply(objects55)
        
        face43.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects56 = [NXOpen.DisplayableObject.Null] * 1 
        face44 = swept2.FindObject("FACE 10019 {(267.9972351707758,3.0415417720583,-0.2045548563002) SWEPT(5)}")
        objects56[0] = face44
        displayModification2.Apply(objects56)
        
        face44.Color = 32767
        
        theSession.DeleteUndoMark(markId80, None)
        
        theSession.SetUndoMarkName(markId71, "扫掠")
        
        sweptBuilder2.Destroy()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression35)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression32)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression33)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression34)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        markId81 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects57 = [NXOpen.DisplayableObject.Null] * 1 
        body2 = workPart.Bodies.FindObject("SWEPT(5)")
        objects57[0] = body2
        theSession.DisplayManager.BlankObjects(objects57)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
    
        # ----------------------------------------------
        #   菜单：插入(S)->组合(B)->合并(U)...
        # ----------------------------------------------
        markId82 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        booleanBuilder1 = workPart.Features.CreateBooleanBuilderUsingCollector(NXOpen.Features.BooleanFeature.Null)
        
        scCollector1 = booleanBuilder1.ToolBodyCollector
        
        scCollector2 = booleanBuilder1.TargetBodyCollector
        
        booleanRegionSelect1 = booleanBuilder1.BooleanRegionSelect
        
        booleanBuilder1.Tolerance = 0.001
        
        scCollector3 = booleanBuilder1.TargetBodyCollector
        
        booleanBuilder1.Operation = NXOpen.Features.Feature.BooleanType.Unite
        
        theSession.SetUndoMarkName(markId82, "合并 对话框")
        
        scCollector4 = workPart.ScCollectors.CreateCollector()
        
        selectionIntentRuleOptions8 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions8.SetSelectedFromInactive(False)
        
        bodies1 = [NXOpen.Body.Null] * 1 
        bodies1[0] = body1
        bodyDumbRule1 = workPart.ScRuleFactory.CreateRuleBodyDumb(bodies1, True, selectionIntentRuleOptions8)
        
        selectionIntentRuleOptions8.Dispose()
        rules8 = [None] * 1 
        rules8[0] = bodyDumbRule1
        scCollector4.ReplaceRules(rules8, False)
        
        booleanBuilder1.TargetBodyCollector = scCollector4
        
        targets1 = [NXOpen.TaggedObject.Null] * 1 
        targets1[0] = body1
        booleanRegionSelect1.AssignTargets(targets1)
        
        scCollector5 = workPart.ScCollectors.CreateCollector()
        
        selectionIntentRuleOptions9 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions9.SetSelectedFromInactive(False)
        
        features5 = [NXOpen.Features.Feature.Null] * 1 
        features5[0] = swept2
        bodyFeatureRule1 = workPart.ScRuleFactory.CreateRuleBodyFeature(features5, False, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions9)
        
        selectionIntentRuleOptions9.Dispose()
        rules9 = [None] * 1 
        rules9[0] = bodyFeatureRule1
        scCollector5.ReplaceRules(rules9, False)
        
        booleanBuilder1.ToolBodyCollector = scCollector5
        
        targets2 = [NXOpen.TaggedObject.Null] * 1 
        targets2[0] = body1
        booleanRegionSelect1.AssignTargets(targets2)
        
        booleanBuilder1.CopyTargets = True
        
        booleanBuilder1.CopyTools = True
        
        markId83 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "合并")
        
        theSession.DeleteUndoMark(markId83, None)
        
        markId84 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "合并")
        
        nXObject13 = booleanBuilder1.Commit()
        
        theSession.DeleteUndoMark(markId84, None)
        
        theSession.SetUndoMarkName(markId82, "合并")
        
        booleanBuilder1.Destroy()
        
        markId85 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects58 = [NXOpen.DisplayableObject.Null] * 1 
        body3 = workPart.Bodies.FindObject("UNITE(6)")
        objects58[0] = body3
        theSession.DisplayManager.BlankObjects(objects58)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：分析(L)  ->测量(S)...
        # ----------------------------------------------
        markId86 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        theSession.SetUndoMarkName(markId86, "测量 对话框")
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector6 = workPart.ScCollectors.CreateCollector()
        
        scCollector6.SetMultiComponent()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        selectionIntentRuleOptions10 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions10.SetSelectedFromInactive(False)
        
        features6 = [NXOpen.Features.Feature.Null] * 1 
        features6[0] = swept1
        bodyFeatureRule2 = workPart.ScRuleFactory.CreateRuleBodyFeature(features6, True, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions10)
        
        selectionIntentRuleOptions10.Dispose()
        rules10 = [None] * 1 
        rules10[0] = bodyFeatureRule2
        scCollector6.ReplaceRules(rules10, False)
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector7 = workPart.ScCollectors.CreateCollector()
        
        scCollector7.SetMultiComponent()
        
        markId87 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        theSession.DeleteUndoMark(markId87, None)
        
        markId88 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        markId89 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Measurement Apply")
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        measureMaster1 = workPart.MeasureManager.MasterMeasurement()
        
        measureMaster1.SequenceType = NXOpen.MeasureMaster.Sequence.Free
        
        measureMaster1.UpdateAtTimestamp = True
        
        measureMaster1.SetNameSuffix("实体")
        
        massUnits1 = [NXOpen.Unit.Null] * 8 
        massUnits1[0] = unit1
        unit3 = workPart.UnitCollection.FindObject("SquareMilliMeter")
        massUnits1[1] = unit3
        unit4 = workPart.UnitCollection.FindObject("CubicMilliMeter")
        massUnits1[2] = unit4
        unit5 = workPart.UnitCollection.FindObject("KilogramPerCubicMilliMeter")
        massUnits1[3] = unit5
        unit6 = workPart.UnitCollection.FindObject("Kilogram")
        massUnits1[4] = unit6
        unit7 = workPart.UnitCollection.FindObject("KilogramMilliMeterSquared")
        massUnits1[5] = unit7
        unit8 = workPart.UnitCollection.FindObject("KilogramMilliMeter")
        massUnits1[6] = unit8
        unit9 = workPart.UnitCollection.FindObject("Newton")
        massUnits1[7] = unit9
        measureElement1 = workPart.MeasureManager.BodyElement(measureMaster1, massUnits1, 0.98999999999999999, scCollector6)
        
        measureElement1.MeasureObject1 = NXOpen.MeasureElement.Measure.Object
        
        measureElement1.SingleSelect1 = True
        
        measureElement1.SetExpressionState(0, False)
        
        measureElement1.SetGeometryState(0, False)
        
        measureElement1.SetAnnotationState(0, False)
        
        measureElement1.SetApproximateState(0, False)
        
        measureElement1.SetExpressionState(1, True)
        
        measureElement1.SetGeometryState(1, False)
        
        measureElement1.SetAnnotationState(1, True)
        
        measureElement1.SetApproximateState(1, False)
        
        measureElement1.SetExpressionState(2, False)
        
        measureElement1.SetGeometryState(2, False)
        
        measureElement1.SetAnnotationState(2, False)
        
        measureElement1.SetApproximateState(2, False)
        
        measureElement1.SetExpressionState(3, False)
        
        measureElement1.SetGeometryState(3, False)
        
        measureElement1.SetAnnotationState(3, False)
        
        measureElement1.SetApproximateState(3, False)
        
        measureElement1.SetExpressionState(4, False)
        
        measureElement1.SetGeometryState(4, False)
        
        measureElement1.SetAnnotationState(4, False)
        
        measureElement1.SetApproximateState(4, False)
        
        measureElement1.SetExpressionState(5, False)
        
        measureElement1.SetGeometryState(5, False)
        
        measureElement1.SetAnnotationState(5, False)
        
        measureElement1.SetApproximateState(5, False)
        
        measureElement1.SetExpressionState(6, False)
        
        measureElement1.SetGeometryState(6, False)
        
        measureElement1.SetAnnotationState(6, False)
        
        measureElement1.SetApproximateState(6, False)
        
        measureElement1.SetExpressionState(7, False)
        
        measureElement1.SetGeometryState(7, False)
        
        measureElement1.SetAnnotationState(7, False)
        
        measureElement1.SetApproximateState(7, False)
        
        measureElement1.SetExpressionState(8, False)
        
        measureElement1.SetGeometryState(8, False)
        
        measureElement1.SetAnnotationState(8, False)
        
        measureElement1.SetApproximateState(8, False)
        
        measureElement1.SetExpressionState(9, False)
        
        measureElement1.SetGeometryState(9, False)
        
        measureElement1.SetAnnotationState(9, False)
        
        measureElement1.SetApproximateState(9, False)
        
        measureElement1.SetExpressionState(10, False)
        
        measureElement1.SetGeometryState(10, False)
        
        measureElement1.SetAnnotationState(10, False)
        
        measureElement1.SetApproximateState(10, False)
        
        measureElement1.SetExpressionState(11, False)
        
        measureElement1.SetGeometryState(11, False)
        
        measureElement1.SetAnnotationState(11, False)
        
        measureElement1.SetApproximateState(11, False)
        
        measureElement1.SetExpressionState(12, False)
        
        measureElement1.SetGeometryState(12, False)
        
        measureElement1.SetAnnotationState(12, False)
        
        measureElement1.SetApproximateState(12, False)
        
        measureElement1.SetExpressionState(13, False)
        
        measureElement1.SetGeometryState(13, False)
        
        measureElement1.SetAnnotationState(13, False)
        
        measureElement1.SetApproximateState(13, False)
        
        measureElement1.SetExpressionState(14, False)
        
        measureElement1.SetGeometryState(14, False)
        
        measureElement1.SetAnnotationState(14, False)
        
        measureElement1.SetApproximateState(14, False)
        
        measureElement1.SetExpressionState(15, False)
        
        measureElement1.SetGeometryState(15, False)
        
        measureElement1.SetAnnotationState(15, False)
        
        measureElement1.SetApproximateState(15, False)
        
        measureElement1.SetExpressionState(16, False)
        
        measureElement1.SetGeometryState(16, False)
        
        measureElement1.SetAnnotationState(16, False)
        
        measureElement1.SetApproximateState(16, False)
        
        measureElement1.SetExpressionState(17, False)
        
        measureElement1.SetGeometryState(17, False)
        
        measureElement1.SetAnnotationState(17, False)
        
        measureElement1.SetApproximateState(17, False)
        
        measureElement1.SetExpressionState(18, False)
        
        measureElement1.SetGeometryState(18, False)
        
        measureElement1.SetAnnotationState(18, False)
        
        measureElement1.SetApproximateState(18, False)
        
        measureElement1.SetExpressionState(19, False)
        
        measureElement1.SetGeometryState(19, False)
        
        measureElement1.SetAnnotationState(19, False)
        
        measureElement1.SetApproximateState(19, False)
        
        measureElement1.SetExpressionState(20, False)
        
        measureElement1.SetGeometryState(20, False)
        
        measureElement1.SetAnnotationState(20, False)
        
        measureElement1.SetApproximateState(20, False)
        
        measureElement1.SetExpressionState(21, False)
        
        measureElement1.SetGeometryState(21, False)
        
        measureElement1.SetAnnotationState(21, False)
        
        measureElement1.SetApproximateState(21, False)
        
        measureElement1.SetExpressionState(22, False)
        
        measureElement1.SetGeometryState(22, False)
        
        measureElement1.SetAnnotationState(22, False)
        
        measureElement1.SetApproximateState(22, False)
        
        measureElement1.SetExpressionState(23, False)
        
        measureElement1.SetGeometryState(23, False)
        
        measureElement1.SetAnnotationState(23, False)
        
        measureElement1.SetApproximateState(23, False)
        
        measureElement1.SetExpressionState(24, False)
        
        measureElement1.SetGeometryState(24, False)
        
        measureElement1.SetAnnotationState(24, False)
        
        measureElement1.SetApproximateState(24, False)
        
        measureElement1.SetExpressionState(25, False)
        
        measureElement1.SetGeometryState(25, False)
        
        measureElement1.SetAnnotationState(25, False)
        
        measureElement1.SetApproximateState(25, False)
        
        measureElement2 = measureMaster1.GetMeasureElement(0)
        
        measureElement2.CreateGeometry()
        
        measureElement3 = measureMaster1.GetMeasureElement(0)
        
        annotation1 = measureElement3.CreateAnnotation()
        
        measureElement4 = measureMaster1.GetMeasureElement(0)
        
        measureElement5 = measureMaster1.GetMeasureElement(0)
        
        measureElement5.EditAnnotation()
        
        measureMaster1.FixupModelingParents()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        markId90 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Measurement Update")
        
        nErrs10 = theSession.UpdateManager.DoUpdate(markId90)
        
        theSession.DeleteUndoMark(markId90, "Measurement Update")
        
        theSession.DeleteUndoMark(markId89, "Measurement Apply")
        
        datadeleted1 = theSession.DeleteTransientDynamicSectionCutData()
        
        theSession.DeleteUndoMark(markId88, None)
        
        theSession.SetUndoMarkName(markId86, "测量")
        
        scCollector7.Destroy()
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        # ----------------------------------------------
        #   菜单：分析(L)  ->测量(S)...
        # ----------------------------------------------
        markId91 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        theSession.SetUndoMarkName(markId91, "测量 对话框")
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector8 = workPart.ScCollectors.CreateCollector()
        
        scCollector8.SetMultiComponent()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        selectionIntentRuleOptions11 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions11.SetSelectedFromInactive(False)
        
        features7 = [NXOpen.Features.Feature.Null] * 1 
        features7[0] = swept2
        bodyFeatureRule3 = workPart.ScRuleFactory.CreateRuleBodyFeature(features7, True, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions11)
        
        selectionIntentRuleOptions11.Dispose()
        rules11 = [None] * 1 
        rules11[0] = bodyFeatureRule3
        scCollector8.ReplaceRules(rules11, False)
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector9 = workPart.ScCollectors.CreateCollector()
        
        scCollector9.SetMultiComponent()
        
        markId92 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        theSession.DeleteUndoMark(markId92, None)
        
        markId93 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        markId94 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Measurement Apply")
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        measureMaster2 = workPart.MeasureManager.MasterMeasurement()
        
        measureMaster2.SequenceType = NXOpen.MeasureMaster.Sequence.Free
        
        measureMaster2.UpdateAtTimestamp = True
        
        measureMaster2.SetNameSuffix("实体")
        
        massUnits2 = [NXOpen.Unit.Null] * 8 
        massUnits2[0] = unit1
        massUnits2[1] = unit3
        massUnits2[2] = unit4
        massUnits2[3] = unit5
        massUnits2[4] = unit6
        massUnits2[5] = unit7
        massUnits2[6] = unit8
        massUnits2[7] = unit9
        measureElement6 = workPart.MeasureManager.BodyElement(measureMaster2, massUnits2, 0.98999999999999999, scCollector8)
        
        measureElement6.MeasureObject1 = NXOpen.MeasureElement.Measure.Object
        
        measureElement6.SingleSelect1 = True
        
        measureElement6.SetExpressionState(0, False)
        
        measureElement6.SetGeometryState(0, False)
        
        measureElement6.SetAnnotationState(0, False)
        
        measureElement6.SetApproximateState(0, False)
        
        measureElement6.SetExpressionState(1, True)
        
        measureElement6.SetGeometryState(1, False)
        
        measureElement6.SetAnnotationState(1, True)
        
        measureElement6.SetApproximateState(1, False)
        
        measureElement6.SetExpressionState(2, False)
        
        measureElement6.SetGeometryState(2, False)
        
        measureElement6.SetAnnotationState(2, False)
        
        measureElement6.SetApproximateState(2, False)
        
        measureElement6.SetExpressionState(3, False)
        
        measureElement6.SetGeometryState(3, False)
        
        measureElement6.SetAnnotationState(3, False)
        
        measureElement6.SetApproximateState(3, False)
        
        measureElement6.SetExpressionState(4, False)
        
        measureElement6.SetGeometryState(4, False)
        
        measureElement6.SetAnnotationState(4, False)
        
        measureElement6.SetApproximateState(4, False)
        
        measureElement6.SetExpressionState(5, False)
        
        measureElement6.SetGeometryState(5, False)
        
        measureElement6.SetAnnotationState(5, False)
        
        measureElement6.SetApproximateState(5, False)
        
        measureElement6.SetExpressionState(6, False)
        
        measureElement6.SetGeometryState(6, False)
        
        measureElement6.SetAnnotationState(6, False)
        
        measureElement6.SetApproximateState(6, False)
        
        measureElement6.SetExpressionState(7, False)
        
        measureElement6.SetGeometryState(7, False)
        
        measureElement6.SetAnnotationState(7, False)
        
        measureElement6.SetApproximateState(7, False)
        
        measureElement6.SetExpressionState(8, False)
        
        measureElement6.SetGeometryState(8, False)
        
        measureElement6.SetAnnotationState(8, False)
        
        measureElement6.SetApproximateState(8, False)
        
        measureElement6.SetExpressionState(9, False)
        
        measureElement6.SetGeometryState(9, False)
        
        measureElement6.SetAnnotationState(9, False)
        
        measureElement6.SetApproximateState(9, False)
        
        measureElement6.SetExpressionState(10, False)
        
        measureElement6.SetGeometryState(10, False)
        
        measureElement6.SetAnnotationState(10, False)
        
        measureElement6.SetApproximateState(10, False)
        
        measureElement6.SetExpressionState(11, False)
        
        measureElement6.SetGeometryState(11, False)
        
        measureElement6.SetAnnotationState(11, False)
        
        measureElement6.SetApproximateState(11, False)
        
        measureElement6.SetExpressionState(12, False)
        
        measureElement6.SetGeometryState(12, False)
        
        measureElement6.SetAnnotationState(12, False)
        
        measureElement6.SetApproximateState(12, False)
        
        measureElement6.SetExpressionState(13, False)
        
        measureElement6.SetGeometryState(13, False)
        
        measureElement6.SetAnnotationState(13, False)
        
        measureElement6.SetApproximateState(13, False)
        
        measureElement6.SetExpressionState(14, False)
        
        measureElement6.SetGeometryState(14, False)
        
        measureElement6.SetAnnotationState(14, False)
        
        measureElement6.SetApproximateState(14, False)
        
        measureElement6.SetExpressionState(15, False)
        
        measureElement6.SetGeometryState(15, False)
        
        measureElement6.SetAnnotationState(15, False)
        
        measureElement6.SetApproximateState(15, False)
        
        measureElement6.SetExpressionState(16, False)
        
        measureElement6.SetGeometryState(16, False)
        
        measureElement6.SetAnnotationState(16, False)
        
        measureElement6.SetApproximateState(16, False)
        
        measureElement6.SetExpressionState(17, False)
        
        measureElement6.SetGeometryState(17, False)
        
        measureElement6.SetAnnotationState(17, False)
        
        measureElement6.SetApproximateState(17, False)
        
        measureElement6.SetExpressionState(18, False)
        
        measureElement6.SetGeometryState(18, False)
        
        measureElement6.SetAnnotationState(18, False)
        
        measureElement6.SetApproximateState(18, False)
        
        measureElement6.SetExpressionState(19, False)
        
        measureElement6.SetGeometryState(19, False)
        
        measureElement6.SetAnnotationState(19, False)
        
        measureElement6.SetApproximateState(19, False)
        
        measureElement6.SetExpressionState(20, False)
        
        measureElement6.SetGeometryState(20, False)
        
        measureElement6.SetAnnotationState(20, False)
        
        measureElement6.SetApproximateState(20, False)
        
        measureElement6.SetExpressionState(21, False)
        
        measureElement6.SetGeometryState(21, False)
        
        measureElement6.SetAnnotationState(21, False)
        
        measureElement6.SetApproximateState(21, False)
        
        measureElement6.SetExpressionState(22, False)
        
        measureElement6.SetGeometryState(22, False)
        
        measureElement6.SetAnnotationState(22, False)
        
        measureElement6.SetApproximateState(22, False)
        
        measureElement6.SetExpressionState(23, False)
        
        measureElement6.SetGeometryState(23, False)
        
        measureElement6.SetAnnotationState(23, False)
        
        measureElement6.SetApproximateState(23, False)
        
        measureElement6.SetExpressionState(24, False)
        
        measureElement6.SetGeometryState(24, False)
        
        measureElement6.SetAnnotationState(24, False)
        
        measureElement6.SetApproximateState(24, False)
        
        measureElement6.SetExpressionState(25, False)
        
        measureElement6.SetGeometryState(25, False)
        
        measureElement6.SetAnnotationState(25, False)
        
        measureElement6.SetApproximateState(25, False)
        
        measureElement7 = measureMaster2.GetMeasureElement(0)
        
        measureElement7.CreateGeometry()
        
        measureElement8 = measureMaster2.GetMeasureElement(0)
        
        annotation2 = measureElement8.CreateAnnotation()
        
        measureElement9 = measureMaster2.GetMeasureElement(0)
        
        measureElement10 = measureMaster2.GetMeasureElement(0)
        
        measureElement10.EditAnnotation()
        
        measureMaster2.FixupModelingParents()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        markId95 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Measurement Update")
        
        nErrs11 = theSession.UpdateManager.DoUpdate(markId95)
        
        theSession.DeleteUndoMark(markId95, "Measurement Update")
        
        theSession.DeleteUndoMark(markId94, "Measurement Apply")
        
        datadeleted2 = theSession.DeleteTransientDynamicSectionCutData()
        
        theSession.DeleteUndoMark(markId93, None)
        
        theSession.SetUndoMarkName(markId91, "测量")
        
        scCollector9.Destroy()
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        # ----------------------------------------------
        #   菜单：分析(L)  ->测量(S)...
        # ----------------------------------------------
        markId96 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        theSession.SetUndoMarkName(markId96, "测量 对话框")
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector10 = workPart.ScCollectors.CreateCollector()
        
        scCollector10.SetMultiComponent()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        selectionIntentRuleOptions12 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions12.SetSelectedFromInactive(False)
        
        features8 = [NXOpen.Features.Feature.Null] * 1 
        booleanFeature1 = nXObject13
        features8[0] = booleanFeature1
        bodyFeatureRule4 = workPart.ScRuleFactory.CreateRuleBodyFeature(features8, True, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions12)
        
        selectionIntentRuleOptions12.Dispose()
        rules12 = [None] * 1 
        rules12[0] = bodyFeatureRule4
        scCollector10.ReplaceRules(rules12, False)
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector11 = workPart.ScCollectors.CreateCollector()
        
        scCollector11.SetMultiComponent()
        
        markId97 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        theSession.DeleteUndoMark(markId97, None)
        
        markId98 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        markId99 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Measurement Apply")
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        measureMaster3 = workPart.MeasureManager.MasterMeasurement()
        
        measureMaster3.SequenceType = NXOpen.MeasureMaster.Sequence.Free
        
        measureMaster3.UpdateAtTimestamp = True
        
        measureMaster3.SetNameSuffix("实体")
        
        massUnits3 = [NXOpen.Unit.Null] * 8 
        massUnits3[0] = unit1
        massUnits3[1] = unit3
        massUnits3[2] = unit4
        massUnits3[3] = unit5
        massUnits3[4] = unit6
        massUnits3[5] = unit7
        massUnits3[6] = unit8
        massUnits3[7] = unit9
        measureElement11 = workPart.MeasureManager.BodyElement(measureMaster3, massUnits3, 0.98999999999999999, scCollector10)
        
        measureElement11.MeasureObject1 = NXOpen.MeasureElement.Measure.Object
        
        measureElement11.SingleSelect1 = True
        
        measureElement11.SetExpressionState(0, False)
        
        measureElement11.SetGeometryState(0, False)
        
        measureElement11.SetAnnotationState(0, False)
        
        measureElement11.SetApproximateState(0, False)
        
        measureElement11.SetExpressionState(1, True)
        
        measureElement11.SetGeometryState(1, False)
        
        measureElement11.SetAnnotationState(1, True)
        
        measureElement11.SetApproximateState(1, False)
        
        measureElement11.SetExpressionState(2, False)
        
        measureElement11.SetGeometryState(2, False)
        
        measureElement11.SetAnnotationState(2, False)
        
        measureElement11.SetApproximateState(2, False)
        
        measureElement11.SetExpressionState(3, False)
        
        measureElement11.SetGeometryState(3, False)
        
        measureElement11.SetAnnotationState(3, False)
        
        measureElement11.SetApproximateState(3, False)
        
        measureElement11.SetExpressionState(4, False)
        
        measureElement11.SetGeometryState(4, False)
        
        measureElement11.SetAnnotationState(4, False)
        
        measureElement11.SetApproximateState(4, False)
        
        measureElement11.SetExpressionState(5, False)
        
        measureElement11.SetGeometryState(5, False)
        
        measureElement11.SetAnnotationState(5, False)
        
        measureElement11.SetApproximateState(5, False)
        
        measureElement11.SetExpressionState(6, False)
        
        measureElement11.SetGeometryState(6, False)
        
        measureElement11.SetAnnotationState(6, False)
        
        measureElement11.SetApproximateState(6, False)
        
        measureElement11.SetExpressionState(7, False)
        
        measureElement11.SetGeometryState(7, False)
        
        measureElement11.SetAnnotationState(7, False)
        
        measureElement11.SetApproximateState(7, False)
        
        measureElement11.SetExpressionState(8, False)
        
        measureElement11.SetGeometryState(8, False)
        
        measureElement11.SetAnnotationState(8, False)
        
        measureElement11.SetApproximateState(8, False)
        
        measureElement11.SetExpressionState(9, False)
        
        measureElement11.SetGeometryState(9, False)
        
        measureElement11.SetAnnotationState(9, False)
        
        measureElement11.SetApproximateState(9, False)
        
        measureElement11.SetExpressionState(10, False)
        
        measureElement11.SetGeometryState(10, False)
        
        measureElement11.SetAnnotationState(10, False)
        
        measureElement11.SetApproximateState(10, False)
        
        measureElement11.SetExpressionState(11, False)
        
        measureElement11.SetGeometryState(11, False)
        
        measureElement11.SetAnnotationState(11, False)
        
        measureElement11.SetApproximateState(11, False)
        
        measureElement11.SetExpressionState(12, False)
        
        measureElement11.SetGeometryState(12, False)
        
        measureElement11.SetAnnotationState(12, False)
        
        measureElement11.SetApproximateState(12, False)
        
        measureElement11.SetExpressionState(13, False)
        
        measureElement11.SetGeometryState(13, False)
        
        measureElement11.SetAnnotationState(13, False)
        
        measureElement11.SetApproximateState(13, False)
        
        measureElement11.SetExpressionState(14, False)
        
        measureElement11.SetGeometryState(14, False)
        
        measureElement11.SetAnnotationState(14, False)
        
        measureElement11.SetApproximateState(14, False)
        
        measureElement11.SetExpressionState(15, False)
        
        measureElement11.SetGeometryState(15, False)
        
        measureElement11.SetAnnotationState(15, False)
        
        measureElement11.SetApproximateState(15, False)
        
        measureElement11.SetExpressionState(16, False)
        
        measureElement11.SetGeometryState(16, False)
        
        measureElement11.SetAnnotationState(16, False)
        
        measureElement11.SetApproximateState(16, False)
        
        measureElement11.SetExpressionState(17, False)
        
        measureElement11.SetGeometryState(17, False)
        
        measureElement11.SetAnnotationState(17, False)
        
        measureElement11.SetApproximateState(17, False)
        
        measureElement11.SetExpressionState(18, False)
        
        measureElement11.SetGeometryState(18, False)
        
        measureElement11.SetAnnotationState(18, False)
        
        measureElement11.SetApproximateState(18, False)
        
        measureElement11.SetExpressionState(19, False)
        
        measureElement11.SetGeometryState(19, False)
        
        measureElement11.SetAnnotationState(19, False)
        
        measureElement11.SetApproximateState(19, False)
        
        measureElement11.SetExpressionState(20, False)
        
        measureElement11.SetGeometryState(20, False)
        
        measureElement11.SetAnnotationState(20, False)
        
        measureElement11.SetApproximateState(20, False)
        
        measureElement11.SetExpressionState(21, False)
        
        measureElement11.SetGeometryState(21, False)
        
        measureElement11.SetAnnotationState(21, False)
        
        measureElement11.SetApproximateState(21, False)
        
        measureElement11.SetExpressionState(22, False)
        
        measureElement11.SetGeometryState(22, False)
        
        measureElement11.SetAnnotationState(22, False)
        
        measureElement11.SetApproximateState(22, False)
        
        measureElement11.SetExpressionState(23, False)
        
        measureElement11.SetGeometryState(23, False)
        
        measureElement11.SetAnnotationState(23, False)
        
        measureElement11.SetApproximateState(23, False)
        
        measureElement11.SetExpressionState(24, False)
        
        measureElement11.SetGeometryState(24, False)
        
        measureElement11.SetAnnotationState(24, False)
        
        measureElement11.SetApproximateState(24, False)
        
        measureElement11.SetExpressionState(25, False)
        
        measureElement11.SetGeometryState(25, False)
        
        measureElement11.SetAnnotationState(25, False)
        
        measureElement11.SetApproximateState(25, False)
        
        measureElement12 = measureMaster3.GetMeasureElement(0)
        
        measureElement12.CreateGeometry()
        
        measureElement13 = measureMaster3.GetMeasureElement(0)
        
        annotation3 = measureElement13.CreateAnnotation()
        
        measureElement14 = measureMaster3.GetMeasureElement(0)
        
        measureElement15 = measureMaster3.GetMeasureElement(0)
        
        measureElement15.EditAnnotation()
        
        measureMaster3.FixupModelingParents()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        markId100 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Measurement Update")
        
        nErrs12 = theSession.UpdateManager.DoUpdate(markId100)
        
        theSession.DeleteUndoMark(markId100, "Measurement Update")
        
        theSession.DeleteUndoMark(markId99, "Measurement Apply")
        
        datadeleted3 = theSession.DeleteTransientDynamicSectionCutData()
        
        theSession.DeleteUndoMark(markId98, None)
        
        theSession.SetUndoMarkName(markId96, "测量")
        
        scCollector11.Destroy()
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        markId101 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects59 = [NXOpen.DisplayableObject.Null] * 1 
        generalLabel1 = annotation1
        objects59[0] = generalLabel1
        theSession.DisplayManager.BlankObjects(objects59)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        markId102 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects60 = [NXOpen.DisplayableObject.Null] * 1 
        generalLabel2 = annotation2
        objects60[0] = generalLabel2
        theSession.DisplayManager.BlankObjects(objects60)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        markId103 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects61 = [NXOpen.DisplayableObject.Null] * 1 
        generalLabel3 = annotation3
        objects61[0] = generalLabel3
        theSession.DisplayManager.BlankObjects(objects61)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)

        springback_strip_volume = float(generalLabel1.GetText()[0].split()[1])
        print(springback_strip_volume)

        prediction_strip_volume = float(generalLabel2.GetText()[0].split()[1])
        print(prediction_strip_volume)

        union_strip_volume = float(generalLabel3.GetText()[0].split()[1])
        print(union_strip_volume)

        intersection_strip_volume = springback_strip_volume + prediction_strip_volume - union_strip_volume
        print(intersection_strip_volume)

        iou_3d = intersection_strip_volume / union_strip_volume
        print(iou_3d)

        # ----------------------------------------------
        #   菜单：文件(F)->保存(S)
        # ----------------------------------------------
        partSaveStatus1 = workPart.Save(NXOpen.BasePart.SaveComponents.TrueValue, NXOpen.BasePart.CloseAfterSave.FalseValue)
        
        partSaveStatus1.Dispose()
        partCloseResponses1 = theSession.Parts.NewPartCloseResponses()
        
        workPart.Close(NXOpen.BasePart.CloseWholeTree.FalseValue, NXOpen.BasePart.CloseModified.UseResponses, partCloseResponses1)
        
        workPart = NXOpen.Part.Null
        displayPart = NXOpen.Part.Null
        partCloseResponses1.Dispose()
        theSession.ApplicationSwitchImmediate("UG_APP_NOPART")
        
        # ----------------------------------------------
        #   菜单：工具(T)->操作记录(J)->停止录制(S)
        # ----------------------------------------------

        return iou_3d

    except:
        workPart.Close(NXOpen.BasePart.CloseWholeTree.FalseValue, NXOpen.BasePart.CloseModified.UseResponses, None)
    
        workPart = NXOpen.Part.Null
        displayPart = NXOpen.Part.Null
        theSession.ApplicationSwitchImmediate("UG_APP_NOPART")
        return None

# NX 1953
# Journal created by Joker on Sun Jan 21 20:07:06 2024 中国标准时间


def with_bias(strip_section_stp_path, iou_3d_prt_path, springback_strip_line_path, prediction_line_path, bias):
    try:
        theSession  = NXOpen.Session.GetSession()
        # ----------------------------------------------
        #   菜单：文件(F)->新建(N)...
        # ----------------------------------------------
        markId1 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        fileNew1 = theSession.Parts.FileNew()
        
        theSession.SetUndoMarkName(markId1, "新建 对话框")
        
        markId2 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "新建")
        
        theSession.DeleteUndoMark(markId2, None)
        
        markId3 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "新建")
        
        fileNew1.TemplateFileName = "model-plain-1-mm-template.prt"
        
        fileNew1.UseBlankTemplate = False
        
        fileNew1.ApplicationName = "ModelTemplate"
        
        fileNew1.Units = NXOpen.Part.Units.Millimeters
        
        fileNew1.RelationType = ""
        
        fileNew1.UsesMasterModel = "No"
        
        fileNew1.TemplateType = NXOpen.FileNewTemplateType.Item
        
        fileNew1.TemplatePresentationName = "模型"
        
        fileNew1.ItemType = ""
        
        fileNew1.Specialization = ""
        
        fileNew1.SetCanCreateAltrep(False)
        
        fileNew1.NewFileName = iou_3d_prt_path
        
        fileNew1.MasterFileName = ""
        
        fileNew1.MakeDisplayedPart = True
        
        fileNew1.DisplayPartOption = NXOpen.DisplayPartOption.AllowAdditional
        
        nXObject1 = fileNew1.Commit()
        
        workPart = theSession.Parts.Work
        displayPart = theSession.Parts.Display
        theSession.DeleteUndoMark(markId3, None)
        
        fileNew1.Destroy()
        
        theSession.ApplicationSwitchImmediate("UG_APP_MODELING")
        
        markId4 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects1 = [NXOpen.DisplayableObject.Null] * 8 
        datumAxis1 = workPart.Datums.FindObject("DATUM_CSYS(0) Y axis")
        objects1[0] = datumAxis1
        datumPlane1 = workPart.Datums.FindObject("DATUM_CSYS(0) XY plane")
        objects1[1] = datumPlane1
        datumPlane2 = workPart.Datums.FindObject("DATUM_CSYS(0) YZ plane")
        objects1[2] = datumPlane2
        datumAxis2 = workPart.Datums.FindObject("DATUM_CSYS(0) Z axis")
        objects1[3] = datumAxis2
        datumCsys1 = workPart.Features.FindObject("DATUM_CSYS(0)")
        point1 = datumCsys1.FindObject("POINT 1")
        objects1[4] = point1
        datumAxis3 = workPart.Datums.FindObject("DATUM_CSYS(0) X axis")
        objects1[5] = datumAxis3
        cartesianCoordinateSystem1 = datumCsys1.FindObject("CSYSTEM 1")
        objects1[6] = cartesianCoordinateSystem1
        datumPlane3 = workPart.Datums.FindObject("DATUM_CSYS(0) XZ plane")
        objects1[7] = datumPlane3
        theSession.DisplayManager.BlankObjects(objects1)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：文件(F)->导入(M)->STEP214...
        # ----------------------------------------------
        markId5 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        step214Importer1 = theSession.DexManager.CreateStep214Importer()
        
        step214Importer1.SimplifyGeometry = True
        
        step214Importer1.LayerDefault = 1
        
        step214Importer1.OutputFile = "G:\\PHD2\\Code\\e2e_ablation\\scripts\\iou_3d\\iou_3d_strip_section_type_4.prt"
        
        step214Importer1.SettingsFile = "D:\\NX1953\\NX1953\\step214ug\\step214ug.def"
        
        step214Importer1.ObjectTypes.ProductData = True
        
        step214Importer1.OutputFile = ""
        
        theSession.SetUndoMarkName(markId5, "导入 STEP214 文件 对话框")
        
        step214Importer1.SetMode(NXOpen.BaseImporter.Mode.NativeFileSystem)
        
        step214Importer1.OutputFile = "G:\\PHD2\\Code\\e2e_ablation\\scripts\\iou_3d\\iou_3d_strip_section_type_4.prt"
        
        step214Importer1.InputFile = strip_section_stp_path
        
        markId6 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "导入 STEP214 文件")
        
        theSession.DeleteUndoMark(markId6, None)
        
        markId7 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "导入 STEP214 文件")
        
        step214Importer1.FileOpenFlag = False
        
        step214Importer1.ProcessHoldFlag = True
        
        nXObject2 = step214Importer1.Commit()
        
        theSession.DeleteUndoMark(markId7, None)
        
        theSession.SetUndoMarkName(markId5, "导入 STEP214 文件")
        
        step214Importer1.Destroy()
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->7 抽取几何特征
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->关联复制(A)->抽取几何特征(E)...
        # ----------------------------------------------
        markId8 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        wavePointBuilder1 = workPart.Features.CreateWavePointBuilder(NXOpen.Features.Feature.Null)
        
        waveDatumBuilder1 = workPart.Features.CreateWaveDatumBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder1 = workPart.Features.CreateCompositeCurveBuilder(NXOpen.Features.Feature.Null)
        
        extractFaceBuilder1 = workPart.Features.CreateExtractFaceBuilder(NXOpen.Features.Feature.Null)
        
        mirrorBodyBuilder1 = workPart.Features.CreateMirrorBodyBuilder(NXOpen.Features.Feature.Null)
        
        waveSketchBuilder1 = workPart.Features.CreateWaveSketchBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder1.CurveFitData.Tolerance = 0.001
        
        compositeCurveBuilder1.CurveFitData.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder1.Section.SetAllowRefCrvs(False)
        
        extractFaceBuilder1.FaceOption = NXOpen.Features.ExtractFaceBuilder.FaceOptionType.AdjacentFaces
        
        compositeCurveBuilder1.Associative = False
        
        waveDatumBuilder1.ParentPart = NXOpen.Features.WaveDatumBuilder.ParentPartType.WorkPart
        
        wavePointBuilder1.ParentPart = NXOpen.Features.WavePointBuilder.ParentPartType.WorkPart
        
        extractFaceBuilder1.ParentPart = NXOpen.Features.ExtractFaceBuilder.ParentPartType.WorkPart
        
        mirrorBodyBuilder1.ParentPartType = NXOpen.Features.MirrorBodyBuilder.ParentPart.WorkPart
        
        compositeCurveBuilder1.ParentPart = NXOpen.Features.CompositeCurveBuilder.PartType.WorkPart
        
        waveSketchBuilder1.ParentPart = NXOpen.Features.WaveSketchBuilder.ParentPartType.WorkPart
        
        compositeCurveBuilder1.Associative = False
        
        theSession.SetUndoMarkName(markId8, "抽取几何特征 对话框")
        
        compositeCurveBuilder1.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder1.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder1.Section.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder1.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder1.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder1.Associative = False
        
        compositeCurveBuilder1.FixAtCurrentTimestamp = False
        
        compositeCurveBuilder1.HideOriginal = False
        
        compositeCurveBuilder1.InheritDisplayProperties = False
        
        compositeCurveBuilder1.Section.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.CurvesAndPoints)
        
        # ----------------------------------------------
        #   菜单：编辑(E)->选择(L)->全选(S)
        # ----------------------------------------------
        # Refer to the sample NXOpen application, Selection for "Select All" alternatives.
        markId9 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId10 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions1 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions1.SetSelectedFromInactive(False)
        
        curves1 = [NXOpen.IBaseCurve.Null] * 20 
        line1 = workPart.Lines.FindObject("ENTITY 3 4 1")
        curves1[0] = line1
        line2 = workPart.Lines.FindObject("ENTITY 3 3 1")
        curves1[1] = line2
        line3 = workPart.Lines.FindObject("ENTITY 3 2 1")
        curves1[2] = line3
        line4 = workPart.Lines.FindObject("ENTITY 3 1 1")
        curves1[3] = line4
        arc1 = workPart.Arcs.FindObject("ENTITY 5 16 1")
        curves1[4] = arc1
        arc2 = workPart.Arcs.FindObject("ENTITY 5 15 1")
        curves1[5] = arc2
        arc3 = workPart.Arcs.FindObject("ENTITY 5 14 1")
        curves1[6] = arc3
        arc4 = workPart.Arcs.FindObject("ENTITY 5 13 1")
        curves1[7] = arc4
        arc5 = workPart.Arcs.FindObject("ENTITY 5 12 1")
        curves1[8] = arc5
        arc6 = workPart.Arcs.FindObject("ENTITY 5 11 1")
        curves1[9] = arc6
        arc7 = workPart.Arcs.FindObject("ENTITY 5 10 1")
        curves1[10] = arc7
        arc8 = workPart.Arcs.FindObject("ENTITY 5 9 1")
        curves1[11] = arc8
        arc9 = workPart.Arcs.FindObject("ENTITY 5 8 1")
        curves1[12] = arc9
        arc10 = workPart.Arcs.FindObject("ENTITY 5 7 1")
        curves1[13] = arc10
        arc11 = workPart.Arcs.FindObject("ENTITY 5 6 1")
        curves1[14] = arc11
        arc12 = workPart.Arcs.FindObject("ENTITY 5 5 1")
        curves1[15] = arc12
        arc13 = workPart.Arcs.FindObject("ENTITY 5 4 1")
        curves1[16] = arc13
        arc14 = workPart.Arcs.FindObject("ENTITY 5 3 1")
        curves1[17] = arc14
        arc15 = workPart.Arcs.FindObject("ENTITY 5 2 1")
        curves1[18] = arc15
        arc16 = workPart.Arcs.FindObject("ENTITY 5 1 1")
        curves1[19] = arc16
        curveDumbRule1 = workPart.ScRuleFactory.CreateRuleBaseCurveDumb(curves1, selectionIntentRuleOptions1)
        
        selectionIntentRuleOptions1.Dispose()
        compositeCurveBuilder1.Section.AllowSelfIntersection(False)
        
        compositeCurveBuilder1.Section.AllowDegenerateCurves(False)
        
        rules1 = [None] * 1 
        rules1[0] = curveDumbRule1
        helpPoint1 = NXOpen.Point3d(0.0, 0.0, 0.0)
        compositeCurveBuilder1.Section.AddToSection(rules1, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint1, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId10, None)
        
        unit1 = workPart.UnitCollection.FindObject("MilliMeter")
        expression1 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        theSession.DeleteUndoMark(markId9, None)
        
        markId11 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        theSession.DeleteUndoMark(markId11, None)
        
        markId12 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        nXObject3 = compositeCurveBuilder1.Commit()
        
        theSession.DeleteUndoMark(markId12, None)
        
        theSession.SetUndoMarkName(markId8, "抽取几何特征")
        
        compositeCurveBuilder1.Destroy()
        
        waveDatumBuilder1.Destroy()
        
        wavePointBuilder1.Destroy()
        
        extractFaceBuilder1.Destroy()
        
        mirrorBodyBuilder1.Destroy()
        
        waveSketchBuilder1.Destroy()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression1)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        markId13 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects2 = [NXOpen.DisplayableObject.Null] * 20 
        compositeCurve1 = nXObject3
        arc17 = compositeCurve1.FindObject("CURVE 6 {5 (-1.0944139378456,7.6279025434865,0)}")
        objects2[0] = arc17
        arc18 = compositeCurve1.FindObject("CURVE 13 {5 (-0.1010226598725,0.8091835736803,0)}")
        objects2[1] = arc18
        arc19 = compositeCurve1.FindObject("CURVE 20 {5 (-6.4033978610663,-4.0480882173563,0)}")
        objects2[2] = arc19
        arc20 = compositeCurve1.FindObject("CURVE 2 {5 (-6.4033978610663,5.0070858332345,0)}")
        objects2[3] = arc20
        arc21 = compositeCurve1.FindObject("CURVE 9 {5 (-5.509737842347,1.2690245869524,0)}")
        objects2[4] = arc21
        arc22 = compositeCurve1.FindObject("CURVE 16 {5 (-5.4024807058543,-3.865886977212,0)}")
        objects2[5] = arc22
        arc23 = compositeCurve1.FindObject("CURVE 5 {5 (-1.1536242786091,5.380729066167,0)}")
        objects2[6] = arc23
        arc24 = compositeCurve1.FindObject("CURVE 12 {5 (-1.121806526416,2.2786836180602,-0)}")
        objects2[7] = arc24
        line5 = compositeCurve1.FindObject("CURVE 19 {3 (-5.5266032053615,-7.4968990046894,0)}")
        objects2[8] = line5
        arc25 = compositeCurve1.FindObject("CURVE 1 {5 (-6.8121275264869,0.479498807939,0)}")
        objects2[9] = arc25
        arc26 = compositeCurve1.FindObject("CURVE 8 {5 (-5.4024807058543,4.8248845930902,0)}")
        objects2[10] = arc26
        arc27 = compositeCurve1.FindObject("CURVE 15 {5 (-5.509737842347,-0.3100269710743,0)}")
        objects2[11] = arc27
        arc28 = compositeCurve1.FindObject("CURVE 4 {5 (-0.1185517363572,7.915522791755,0)}")
        objects2[12] = arc28
        arc29 = compositeCurve1.FindObject("CURVE 11 {5 (-1.0423110139851,1.1951999955067,0)}")
        objects2[13] = arc29
        arc30 = compositeCurve1.FindObject("CURVE 18 {5 (-4.7346688211586,-8.3353288585981,0)}")
        objects2[14] = arc30
        arc31 = compositeCurve1.FindObject("CURVE 7 {5 (-3.6745728667969,8.8192034738836,0)}")
        objects2[15] = arc31
        line6 = compositeCurve1.FindObject("CURVE 14 {3 (-3.1052263356571,-0.0291839132899,0)}")
        objects2[16] = line6
        arc32 = compositeCurve1.FindObject("CURVE 3 {5 (-4.0885490681162,9.7485344957746,-0)}")
        objects2[17] = arc32
        line7 = compositeCurve1.FindObject("CURVE 10 {3 (-3.1052263356571,0.9881815291681,0)}")
        objects2[18] = line7
        line8 = compositeCurve1.FindObject("CURVE 17 {3 (-4.5547082893861,-7.1961454248122,0)}")
        objects2[19] = line8
        theSession.DisplayManager.BlankObjects(objects2)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：编辑(E)->选择(L)->全选(S)
        # ----------------------------------------------
        # Refer to the sample NXOpen application, Selection for "Select All" alternatives.
        # ----------------------------------------------
        #   菜单：编辑(E)->删除(D)...
        # ----------------------------------------------
        markId14 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Delete")
        
        theSession.UpdateManager.ClearErrorList()
        
        markId15 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Delete")
        
        objects3 = [NXOpen.TaggedObject.Null] * 20 
        objects3[0] = line1
        objects3[1] = line2
        objects3[2] = line3
        objects3[3] = line4
        objects3[4] = arc1
        objects3[5] = arc2
        objects3[6] = arc3
        objects3[7] = arc4
        objects3[8] = arc5
        objects3[9] = arc6
        objects3[10] = arc7
        objects3[11] = arc8
        objects3[12] = arc9
        objects3[13] = arc10
        objects3[14] = arc11
        objects3[15] = arc12
        objects3[16] = arc13
        objects3[17] = arc14
        objects3[18] = arc15
        objects3[19] = arc16
        nErrs1 = theSession.UpdateManager.AddObjectsToDeleteList(objects3)
        
        id1 = theSession.NewestVisibleUndoMark
        
        nErrs2 = theSession.UpdateManager.DoUpdate(id1)
        
        theSession.DeleteUndoMark(markId14, None)
        
        # ----------------------------------------------
        #   菜单：编辑(E)->移动对象(O)...
        # ----------------------------------------------
        markId16 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        moveObjectBuilder1 = workPart.BaseFeatures.CreateMoveObjectBuilder(NXOpen.Features.MoveObject.Null)
        
        moveObjectBuilder1.TransformMotion.DistanceAngle.OrientXpress.AxisOption = NXOpen.GeometricUtilities.OrientXpressBuilder.Axis.Passive
        
        moveObjectBuilder1.TransformMotion.DistanceAngle.OrientXpress.PlaneOption = NXOpen.GeometricUtilities.OrientXpressBuilder.Plane.Passive
        
        moveObjectBuilder1.TransformMotion.AlongCurveAngle.AlongCurve.IsPercentUsed = True
        
        moveObjectBuilder1.TransformMotion.AlongCurveAngle.AlongCurve.Expression.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.AlongCurveAngle.AlongCurve.Expression.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.OrientXpress.AxisOption = NXOpen.GeometricUtilities.OrientXpressBuilder.Axis.Passive
        
        moveObjectBuilder1.TransformMotion.OrientXpress.PlaneOption = NXOpen.GeometricUtilities.OrientXpressBuilder.Plane.Passive
        
        moveObjectBuilder1.TransformMotion.DeltaEnum = NXOpen.GeometricUtilities.ModlMotion.Delta.ReferenceAcsWorkPart
        
        moveObjectBuilder1.TransformMotion.Option = NXOpen.GeometricUtilities.ModlMotion.Options.Angle
        
        moveObjectBuilder1.TransformMotion.DistanceValue.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.DistanceBetweenPointsDistance.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.RadialDistance.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.Angle.SetFormula("90")
        
        moveObjectBuilder1.TransformMotion.DistanceAngle.Distance.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.DistanceAngle.Angle.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.DeltaXc.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.DeltaYc.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.DeltaZc.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.AlongCurveAngle.AlongCurve.Expression.SetFormula("0")
        
        moveObjectBuilder1.TransformMotion.AlongCurveAngle.AlongCurveAngle.SetFormula("0")
        
        theSession.SetUndoMarkName(markId16, "移动对象 对话框")
        
        xform1 = workPart.Xforms.CreateXform(NXOpen.SmartObject.UpdateOption.WithinModeling, 1.0)
        
        cartesianCoordinateSystem2 = workPart.CoordinateSystems.CreateCoordinateSystem(xform1, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        moveObjectBuilder1.TransformMotion.ToCsys = cartesianCoordinateSystem2
        
        moveObjectBuilder1.TransformMotion.Angle.SetFormula("0")
        
        objects4 = [NXOpen.NXObject.Null] * 20 
        objects4[0] = arc25
        objects4[1] = arc20
        objects4[2] = arc32
        objects4[3] = arc28
        objects4[4] = arc23
        objects4[5] = arc17
        objects4[6] = arc31
        objects4[7] = arc26
        objects4[8] = arc21
        objects4[9] = line7
        objects4[10] = arc29
        objects4[11] = arc24
        objects4[12] = arc18
        objects4[13] = line6
        objects4[14] = arc27
        objects4[15] = arc22
        objects4[16] = line8
        objects4[17] = arc30
        objects4[18] = line5
        objects4[19] = arc19
        added1 = moveObjectBuilder1.ObjectToMoveObject.Add(objects4)
        
        markId17 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "起点")
        
        expression2 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        expression3 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        theSession.SetUndoMarkName(markId17, "矢量 对话框")
        
        # ----------------------------------------------
        #   对话开始 矢量
        # ----------------------------------------------
        origin1 = NXOpen.Point3d(0.0, 0.0, 0.0)
        vector1 = NXOpen.Vector3d(0.0, 1.0, 0.0)
        direction1 = workPart.Directions.CreateDirection(origin1, vector1, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        markId18 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "矢量")
        
        theSession.DeleteUndoMark(markId18, None)
        
        markId19 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "矢量")
        
        theSession.DeleteUndoMark(markId19, None)
        
        theSession.SetUndoMarkName(markId17, "矢量")
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression2)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression3)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        theSession.DeleteUndoMark(markId17, None)
        
        direction1.ProtectFromDelete()
        
        expression4 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        axis1 = workPart.Axes.CreateAxis(NXOpen.Point.Null, direction1, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        moveObjectBuilder1.TransformMotion.AngularAxis = axis1
        
        moveObjectBuilder1.TransformMotion.AngularAxis = axis1
        
        expression5 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        markId20 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "起点")
        
        expression6 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        expression7 = workPart.Expressions.CreateSystemExpressionWithUnits("p16_x=0.00000000000", unit1)
        
        expression8 = workPart.Expressions.CreateSystemExpressionWithUnits("p17_y=0.00000000000", unit1)
        
        expression9 = workPart.Expressions.CreateSystemExpressionWithUnits("p18_z=0.00000000000", unit1)
        
        expression10 = workPart.Expressions.CreateSystemExpressionWithUnits("p19_xdelta=0.00000000000", unit1)
        
        expression11 = workPart.Expressions.CreateSystemExpressionWithUnits("p20_ydelta=0.00000000000", unit1)
        
        expression12 = workPart.Expressions.CreateSystemExpressionWithUnits("p21_zdelta=0.00000000000", unit1)
        
        expression13 = workPart.Expressions.CreateSystemExpressionWithUnits("p22_radius=0.00000000000", unit1)
        
        unit2 = moveObjectBuilder1.TransformMotion.DistanceAngle.Angle.Units
        
        expression14 = workPart.Expressions.CreateSystemExpressionWithUnits("p23_angle=0.00000000000", unit2)
        
        expression15 = workPart.Expressions.CreateSystemExpressionWithUnits("p24_zdelta=0.00000000000", unit1)
        
        expression16 = workPart.Expressions.CreateSystemExpressionWithUnits("p25_radius=0.00000000000", unit1)
        
        expression17 = workPart.Expressions.CreateSystemExpressionWithUnits("p26_angle1=0.00000000000", unit2)
        
        expression18 = workPart.Expressions.CreateSystemExpressionWithUnits("p27_angle2=0.00000000000", unit2)
        
        expression19 = workPart.Expressions.CreateSystemExpressionWithUnits("p28_distance=0", unit1)
        
        expression20 = workPart.Expressions.CreateSystemExpressionWithUnits("p29_arclen=0", unit1)
        
        expression21 = workPart.Expressions.CreateSystemExpressionWithUnits("p30_percent=0", NXOpen.Unit.Null)
        
        expression7.SetFormula("0")
        
        expression8.SetFormula("0")
        
        expression9.SetFormula("0")
        
        expression10.SetFormula("0")
        
        expression11.SetFormula("0")
        
        expression12.SetFormula("0")
        
        expression13.SetFormula("0")
        
        expression14.SetFormula("0")
        
        expression15.SetFormula("0")
        
        expression16.SetFormula("0")
        
        expression17.SetFormula("0")
        
        expression18.SetFormula("0")
        
        expression19.SetFormula("0")
        
        expression21.SetFormula("100")
        
        expression20.SetFormula("0")
        
        theSession.SetUndoMarkName(markId20, "点 对话框")
        
        expression22 = workPart.Expressions.CreateSystemExpressionWithUnits("p31_x=0.00000000000", unit1)
        
        scalar1 = workPart.Scalars.CreateScalarExpression(expression22, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        expression23 = workPart.Expressions.CreateSystemExpressionWithUnits("p32_y=0.00000000000", unit1)
        
        scalar2 = workPart.Scalars.CreateScalarExpression(expression23, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        expression24 = workPart.Expressions.CreateSystemExpressionWithUnits("p33_z=0.00000000000", unit1)
        
        scalar3 = workPart.Scalars.CreateScalarExpression(expression24, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        point2 = workPart.Points.CreatePoint(scalar1, scalar2, scalar3, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        expression7.SetFormula("0.00000000000")
        
        expression8.SetFormula("0.00000000000")
        
        expression9.SetFormula("0.00000000000")
        
        expression10.SetFormula("0.00000000000")
        
        expression11.SetFormula("0.00000000000")
        
        expression12.SetFormula("0.00000000000")
        
        expression13.SetFormula("0.00000000000")
        
        expression14.SetFormula("0.00000000000")
        
        expression15.SetFormula("0.00000000000")
        
        expression16.SetFormula("0.00000000000")
        
        expression17.SetFormula("0.00000000000")
        
        expression18.SetFormula("0.00000000000")
        
        expression21.SetFormula("100.00000000000")
        
        # ----------------------------------------------
        #   对话开始 点
        # ----------------------------------------------
        markId21 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "点")
        
        theSession.DeleteUndoMark(markId21, None)
        
        markId22 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "点")
        
        expression7.RightHandSide = "0.00000000000"
        
        expression8.RightHandSide = "0.00000000000"
        
        expression9.RightHandSide = "0.00000000000"
        
        workPart.Points.DeletePoint(point2)
        
        expression25 = workPart.Expressions.CreateSystemExpressionWithUnits("p17_x=0.00000000000", unit1)
        
        scalar4 = workPart.Scalars.CreateScalarExpression(expression25, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        expression26 = workPart.Expressions.CreateSystemExpressionWithUnits("p18_y=0.00000000000", unit1)
        
        scalar5 = workPart.Scalars.CreateScalarExpression(expression26, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        expression27 = workPart.Expressions.CreateSystemExpressionWithUnits("p19_z=0.00000000000", unit1)
        
        scalar6 = workPart.Scalars.CreateScalarExpression(expression27, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        point3 = workPart.Points.CreatePoint(scalar4, scalar5, scalar6, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        theSession.DeleteUndoMark(markId22, None)
        
        theSession.SetUndoMarkName(markId20, "点")
        
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression7)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression8)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression9)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression10)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression11)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression12)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression13)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression14)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression15)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression16)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression17)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression18)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression19)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression20)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        try:
            # 表达式仍然在使用中。
            workPart.Expressions.Delete(expression21)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1050029)
            
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression6)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        theSession.DeleteUndoMark(markId20, None)
        
        scalar7 = workPart.Scalars.CreateScalarExpression(expression25, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        scalar8 = workPart.Scalars.CreateScalarExpression(expression26, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        scalar9 = workPart.Scalars.CreateScalarExpression(expression27, NXOpen.Scalar.DimensionalityType.NotSet, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        point4 = workPart.Points.CreatePoint(scalar7, scalar8, scalar9, NXOpen.SmartObject.UpdateOption.WithinModeling)
        
        point5 = axis1.Point
        
        axis1.Point = point3
        
        moveObjectBuilder1.TransformMotion.AngularAxis = axis1
        
        moveObjectBuilder1.TransformMotion.Angle.SetFormula("90")
        
        moveObjectBuilder1.TransformMotion.Angle.SetFormula("90")
        
        markId23 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "移动对象")
        
        theSession.DeleteUndoMark(markId23, None)
        
        markId24 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "移动对象")
        
        nXObject4 = moveObjectBuilder1.Commit()
        
        objects5 = moveObjectBuilder1.GetCommittedObjects()
        
        theSession.DeleteUndoMark(markId24, None)
        
        theSession.SetUndoMarkName(markId16, "移动对象")
        
        moveObjectBuilder1.Destroy()
        
        markId25 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "")
        
        nErrs3 = theSession.UpdateManager.DoUpdate(markId25)
        
        theSession.DeleteUndoMark(markId25, "")
        
        direction1.ReleaseDeleteProtection()
        
        workPart.Points.DeletePoint(point4)
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression5)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression4)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        # ----------------------------------------------
        #   菜单：文件(F)->导入(M)->文件中的点(L)...
        # ----------------------------------------------
        markId26 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Import Points from File")
        
        pointsFromFileBuilder1 = workPart.CreatePointsFromFileBuilder()
        
        pointsFromFileBuilder1.FileName = springback_strip_line_path
        
        pointsFromFileBuilder1.CoordinateOption = NXOpen.GeometricUtilities.PointsFromFileBuilder.Options.Absolute
        
        nXObject5 = pointsFromFileBuilder1.Commit()
        
        pointsFromFileBuilder1.Destroy()
        
        nErrs4 = theSession.UpdateManager.DoUpdate(markId26)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->10 拟合曲线
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->曲线(C)->拟合曲线(F)...
        # ----------------------------------------------
        markId27 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        fitCurveBuilder1 = workPart.Features.CreateFitCurveBuilder(NXOpen.Features.FitCurve.Null)
        
        fitCurveBuilder1.Tolerance = 0.001
        
        fitCurveBuilder1.TargetSourceType = NXOpen.Features.FitCurveBuilder.TargetSourceTypes.SpecifiedPoints
        
        fitCurveBuilder1.ProjectionDirectionOption = NXOpen.Features.FitCurveBuilder.ProjectionDirectionOptions.Normal
        
        fitCurveBuilder1.Radius.SetFormula("50")
        
        fitCurveBuilder1.Degree = 24
        
        fitCurveBuilder1.HasUniformSegments = True
        
        fitCurveBuilder1.Extender.StartValue.SetFormula("0")
        
        fitCurveBuilder1.Extender.EndValue.SetFormula("0")
        
        fitCurveBuilder1.RejectionThreshold.SetFormula("10")
        
        fitCurveBuilder1.IsAssociative = False
        
        theSession.SetUndoMarkName(markId27, "拟合曲线 对话框")
        
        markId28 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        theSession.DeleteUndoMark(markId28, None)
        
        markId29 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        # ----------------------------------------------
        #   菜单：编辑(E)->选择(L)->全选(S)
        # ----------------------------------------------
        # Refer to the sample NXOpen application, Selection for "Select All" alternatives.
        objects6 = [NXOpen.TaggedObject.Null] * 302 
        point6 = workPart.Points.FindObject("ENTITY 2 301 1")
        objects6[0] = point6
        point7 = workPart.Points.FindObject("ENTITY 2 300 1")
        objects6[1] = point7
        point8 = workPart.Points.FindObject("ENTITY 2 299 1")
        objects6[2] = point8
        point9 = workPart.Points.FindObject("ENTITY 2 298 1")
        objects6[3] = point9
        point10 = workPart.Points.FindObject("ENTITY 2 297 1")
        objects6[4] = point10
        point11 = workPart.Points.FindObject("ENTITY 2 296 1")
        objects6[5] = point11
        point12 = workPart.Points.FindObject("ENTITY 2 295 1")
        objects6[6] = point12
        point13 = workPart.Points.FindObject("ENTITY 2 294 1")
        objects6[7] = point13
        point14 = workPart.Points.FindObject("ENTITY 2 293 1")
        objects6[8] = point14
        point15 = workPart.Points.FindObject("ENTITY 2 292 1")
        objects6[9] = point15
        point16 = workPart.Points.FindObject("ENTITY 2 291 1")
        objects6[10] = point16
        point17 = workPart.Points.FindObject("ENTITY 2 290 1")
        objects6[11] = point17
        point18 = workPart.Points.FindObject("ENTITY 2 289 1")
        objects6[12] = point18
        point19 = workPart.Points.FindObject("ENTITY 2 288 1")
        objects6[13] = point19
        point20 = workPart.Points.FindObject("ENTITY 2 287 1")
        objects6[14] = point20
        point21 = workPart.Points.FindObject("ENTITY 2 286 1")
        objects6[15] = point21
        point22 = workPart.Points.FindObject("ENTITY 2 285 1")
        objects6[16] = point22
        point23 = workPart.Points.FindObject("ENTITY 2 284 1")
        objects6[17] = point23
        point24 = workPart.Points.FindObject("ENTITY 2 283 1")
        objects6[18] = point24
        point25 = workPart.Points.FindObject("ENTITY 2 282 1")
        objects6[19] = point25
        point26 = workPart.Points.FindObject("ENTITY 2 281 1")
        objects6[20] = point26
        point27 = workPart.Points.FindObject("ENTITY 2 280 1")
        objects6[21] = point27
        point28 = workPart.Points.FindObject("ENTITY 2 279 1")
        objects6[22] = point28
        point29 = workPart.Points.FindObject("ENTITY 2 278 1")
        objects6[23] = point29
        point30 = workPart.Points.FindObject("ENTITY 2 277 1")
        objects6[24] = point30
        point31 = workPart.Points.FindObject("ENTITY 2 276 1")
        objects6[25] = point31
        point32 = workPart.Points.FindObject("ENTITY 2 275 1")
        objects6[26] = point32
        point33 = workPart.Points.FindObject("ENTITY 2 274 1")
        objects6[27] = point33
        point34 = workPart.Points.FindObject("ENTITY 2 273 1")
        objects6[28] = point34
        point35 = workPart.Points.FindObject("ENTITY 2 272 1")
        objects6[29] = point35
        point36 = workPart.Points.FindObject("ENTITY 2 271 1")
        objects6[30] = point36
        point37 = workPart.Points.FindObject("ENTITY 2 270 1")
        objects6[31] = point37
        point38 = workPart.Points.FindObject("ENTITY 2 269 1")
        objects6[32] = point38
        point39 = workPart.Points.FindObject("ENTITY 2 268 1")
        objects6[33] = point39
        point40 = workPart.Points.FindObject("ENTITY 2 267 1")
        objects6[34] = point40
        point41 = workPart.Points.FindObject("ENTITY 2 266 1")
        objects6[35] = point41
        point42 = workPart.Points.FindObject("ENTITY 2 265 1")
        objects6[36] = point42
        point43 = workPart.Points.FindObject("ENTITY 2 264 1")
        objects6[37] = point43
        point44 = workPart.Points.FindObject("ENTITY 2 263 1")
        objects6[38] = point44
        point45 = workPart.Points.FindObject("ENTITY 2 262 1")
        objects6[39] = point45
        point46 = workPart.Points.FindObject("ENTITY 2 261 1")
        objects6[40] = point46
        point47 = workPart.Points.FindObject("ENTITY 2 260 1")
        objects6[41] = point47
        point48 = workPart.Points.FindObject("ENTITY 2 259 1")
        objects6[42] = point48
        point49 = workPart.Points.FindObject("ENTITY 2 258 1")
        objects6[43] = point49
        point50 = workPart.Points.FindObject("ENTITY 2 257 1")
        objects6[44] = point50
        point51 = workPart.Points.FindObject("ENTITY 2 256 1")
        objects6[45] = point51
        point52 = workPart.Points.FindObject("ENTITY 2 255 1")
        objects6[46] = point52
        point53 = workPart.Points.FindObject("ENTITY 2 254 1")
        objects6[47] = point53
        point54 = workPart.Points.FindObject("ENTITY 2 253 1")
        objects6[48] = point54
        point55 = workPart.Points.FindObject("ENTITY 2 252 1")
        objects6[49] = point55
        point56 = workPart.Points.FindObject("ENTITY 2 251 1")
        objects6[50] = point56
        point57 = workPart.Points.FindObject("ENTITY 2 250 1")
        objects6[51] = point57
        point58 = workPart.Points.FindObject("ENTITY 2 249 1")
        objects6[52] = point58
        point59 = workPart.Points.FindObject("ENTITY 2 248 1")
        objects6[53] = point59
        point60 = workPart.Points.FindObject("ENTITY 2 247 1")
        objects6[54] = point60
        point61 = workPart.Points.FindObject("ENTITY 2 246 1")
        objects6[55] = point61
        point62 = workPart.Points.FindObject("ENTITY 2 245 1")
        objects6[56] = point62
        point63 = workPart.Points.FindObject("ENTITY 2 244 1")
        objects6[57] = point63
        point64 = workPart.Points.FindObject("ENTITY 2 243 1")
        objects6[58] = point64
        point65 = workPart.Points.FindObject("ENTITY 2 242 1")
        objects6[59] = point65
        point66 = workPart.Points.FindObject("ENTITY 2 241 1")
        objects6[60] = point66
        point67 = workPart.Points.FindObject("ENTITY 2 240 1")
        objects6[61] = point67
        point68 = workPart.Points.FindObject("ENTITY 2 239 1")
        objects6[62] = point68
        point69 = workPart.Points.FindObject("ENTITY 2 238 1")
        objects6[63] = point69
        point70 = workPart.Points.FindObject("ENTITY 2 237 1")
        objects6[64] = point70
        point71 = workPart.Points.FindObject("ENTITY 2 236 1")
        objects6[65] = point71
        point72 = workPart.Points.FindObject("ENTITY 2 235 1")
        objects6[66] = point72
        point73 = workPart.Points.FindObject("ENTITY 2 234 1")
        objects6[67] = point73
        point74 = workPart.Points.FindObject("ENTITY 2 233 1")
        objects6[68] = point74
        point75 = workPart.Points.FindObject("ENTITY 2 232 1")
        objects6[69] = point75
        point76 = workPart.Points.FindObject("ENTITY 2 231 1")
        objects6[70] = point76
        point77 = workPart.Points.FindObject("ENTITY 2 230 1")
        objects6[71] = point77
        point78 = workPart.Points.FindObject("ENTITY 2 229 1")
        objects6[72] = point78
        point79 = workPart.Points.FindObject("ENTITY 2 228 1")
        objects6[73] = point79
        point80 = workPart.Points.FindObject("ENTITY 2 227 1")
        objects6[74] = point80
        point81 = workPart.Points.FindObject("ENTITY 2 226 1")
        objects6[75] = point81
        point82 = workPart.Points.FindObject("ENTITY 2 225 1")
        objects6[76] = point82
        point83 = workPart.Points.FindObject("ENTITY 2 224 1")
        objects6[77] = point83
        point84 = workPart.Points.FindObject("ENTITY 2 223 1")
        objects6[78] = point84
        point85 = workPart.Points.FindObject("ENTITY 2 222 1")
        objects6[79] = point85
        point86 = workPart.Points.FindObject("ENTITY 2 221 1")
        objects6[80] = point86
        point87 = workPart.Points.FindObject("ENTITY 2 220 1")
        objects6[81] = point87
        point88 = workPart.Points.FindObject("ENTITY 2 219 1")
        objects6[82] = point88
        point89 = workPart.Points.FindObject("ENTITY 2 218 1")
        objects6[83] = point89
        point90 = workPart.Points.FindObject("ENTITY 2 217 1")
        objects6[84] = point90
        point91 = workPart.Points.FindObject("ENTITY 2 216 1")
        objects6[85] = point91
        point92 = workPart.Points.FindObject("ENTITY 2 215 1")
        objects6[86] = point92
        point93 = workPart.Points.FindObject("ENTITY 2 214 1")
        objects6[87] = point93
        point94 = workPart.Points.FindObject("ENTITY 2 213 1")
        objects6[88] = point94
        point95 = workPart.Points.FindObject("ENTITY 2 212 1")
        objects6[89] = point95
        point96 = workPart.Points.FindObject("ENTITY 2 211 1")
        objects6[90] = point96
        point97 = workPart.Points.FindObject("ENTITY 2 210 1")
        objects6[91] = point97
        point98 = workPart.Points.FindObject("ENTITY 2 209 1")
        objects6[92] = point98
        point99 = workPart.Points.FindObject("ENTITY 2 208 1")
        objects6[93] = point99
        point100 = workPart.Points.FindObject("ENTITY 2 207 1")
        objects6[94] = point100
        point101 = workPart.Points.FindObject("ENTITY 2 206 1")
        objects6[95] = point101
        point102 = workPart.Points.FindObject("ENTITY 2 205 1")
        objects6[96] = point102
        point103 = workPart.Points.FindObject("ENTITY 2 204 1")
        objects6[97] = point103
        point104 = workPart.Points.FindObject("ENTITY 2 203 1")
        objects6[98] = point104
        point105 = workPart.Points.FindObject("ENTITY 2 202 1")
        objects6[99] = point105
        point106 = workPart.Points.FindObject("ENTITY 2 201 1")
        objects6[100] = point106
        point107 = workPart.Points.FindObject("ENTITY 2 200 1")
        objects6[101] = point107
        point108 = workPart.Points.FindObject("ENTITY 2 199 1")
        objects6[102] = point108
        point109 = workPart.Points.FindObject("ENTITY 2 198 1")
        objects6[103] = point109
        point110 = workPart.Points.FindObject("ENTITY 2 197 1")
        objects6[104] = point110
        point111 = workPart.Points.FindObject("ENTITY 2 196 1")
        objects6[105] = point111
        point112 = workPart.Points.FindObject("ENTITY 2 195 1")
        objects6[106] = point112
        point113 = workPart.Points.FindObject("ENTITY 2 194 1")
        objects6[107] = point113
        point114 = workPart.Points.FindObject("ENTITY 2 193 1")
        objects6[108] = point114
        point115 = workPart.Points.FindObject("ENTITY 2 192 1")
        objects6[109] = point115
        point116 = workPart.Points.FindObject("ENTITY 2 191 1")
        objects6[110] = point116
        point117 = workPart.Points.FindObject("ENTITY 2 190 1")
        objects6[111] = point117
        point118 = workPart.Points.FindObject("ENTITY 2 189 1")
        objects6[112] = point118
        point119 = workPart.Points.FindObject("ENTITY 2 188 1")
        objects6[113] = point119
        point120 = workPart.Points.FindObject("ENTITY 2 187 1")
        objects6[114] = point120
        point121 = workPart.Points.FindObject("ENTITY 2 186 1")
        objects6[115] = point121
        point122 = workPart.Points.FindObject("ENTITY 2 185 1")
        objects6[116] = point122
        point123 = workPart.Points.FindObject("ENTITY 2 184 1")
        objects6[117] = point123
        point124 = workPart.Points.FindObject("ENTITY 2 183 1")
        objects6[118] = point124
        point125 = workPart.Points.FindObject("ENTITY 2 182 1")
        objects6[119] = point125
        point126 = workPart.Points.FindObject("ENTITY 2 181 1")
        objects6[120] = point126
        point127 = workPart.Points.FindObject("ENTITY 2 180 1")
        objects6[121] = point127
        point128 = workPart.Points.FindObject("ENTITY 2 179 1")
        objects6[122] = point128
        point129 = workPart.Points.FindObject("ENTITY 2 178 1")
        objects6[123] = point129
        point130 = workPart.Points.FindObject("ENTITY 2 177 1")
        objects6[124] = point130
        point131 = workPart.Points.FindObject("ENTITY 2 176 1")
        objects6[125] = point131
        point132 = workPart.Points.FindObject("ENTITY 2 175 1")
        objects6[126] = point132
        point133 = workPart.Points.FindObject("ENTITY 2 174 1")
        objects6[127] = point133
        point134 = workPart.Points.FindObject("ENTITY 2 173 1")
        objects6[128] = point134
        point135 = workPart.Points.FindObject("ENTITY 2 172 1")
        objects6[129] = point135
        point136 = workPart.Points.FindObject("ENTITY 2 171 1")
        objects6[130] = point136
        point137 = workPart.Points.FindObject("ENTITY 2 170 1")
        objects6[131] = point137
        point138 = workPart.Points.FindObject("ENTITY 2 169 1")
        objects6[132] = point138
        point139 = workPart.Points.FindObject("ENTITY 2 168 1")
        objects6[133] = point139
        point140 = workPart.Points.FindObject("ENTITY 2 167 1")
        objects6[134] = point140
        point141 = workPart.Points.FindObject("ENTITY 2 166 1")
        objects6[135] = point141
        point142 = workPart.Points.FindObject("ENTITY 2 165 1")
        objects6[136] = point142
        point143 = workPart.Points.FindObject("ENTITY 2 164 1")
        objects6[137] = point143
        point144 = workPart.Points.FindObject("ENTITY 2 163 1")
        objects6[138] = point144
        point145 = workPart.Points.FindObject("ENTITY 2 162 1")
        objects6[139] = point145
        point146 = workPart.Points.FindObject("ENTITY 2 161 1")
        objects6[140] = point146
        point147 = workPart.Points.FindObject("ENTITY 2 160 1")
        objects6[141] = point147
        point148 = workPart.Points.FindObject("ENTITY 2 159 1")
        objects6[142] = point148
        point149 = workPart.Points.FindObject("ENTITY 2 158 1")
        objects6[143] = point149
        point150 = workPart.Points.FindObject("ENTITY 2 157 1")
        objects6[144] = point150
        point151 = workPart.Points.FindObject("ENTITY 2 156 1")
        objects6[145] = point151
        point152 = workPart.Points.FindObject("ENTITY 2 155 1")
        objects6[146] = point152
        point153 = workPart.Points.FindObject("ENTITY 2 154 1")
        objects6[147] = point153
        point154 = workPart.Points.FindObject("ENTITY 2 153 1")
        objects6[148] = point154
        point155 = workPart.Points.FindObject("ENTITY 2 152 1")
        objects6[149] = point155
        point156 = workPart.Points.FindObject("ENTITY 2 151 1")
        objects6[150] = point156
        point157 = workPart.Points.FindObject("ENTITY 2 150 1")
        objects6[151] = point157
        point158 = workPart.Points.FindObject("ENTITY 2 149 1")
        objects6[152] = point158
        point159 = workPart.Points.FindObject("ENTITY 2 148 1")
        objects6[153] = point159
        point160 = workPart.Points.FindObject("ENTITY 2 147 1")
        objects6[154] = point160
        point161 = workPart.Points.FindObject("ENTITY 2 146 1")
        objects6[155] = point161
        point162 = workPart.Points.FindObject("ENTITY 2 145 1")
        objects6[156] = point162
        point163 = workPart.Points.FindObject("ENTITY 2 144 1")
        objects6[157] = point163
        point164 = workPart.Points.FindObject("ENTITY 2 143 1")
        objects6[158] = point164
        point165 = workPart.Points.FindObject("ENTITY 2 142 1")
        objects6[159] = point165
        point166 = workPart.Points.FindObject("ENTITY 2 141 1")
        objects6[160] = point166
        point167 = workPart.Points.FindObject("ENTITY 2 140 1")
        objects6[161] = point167
        point168 = workPart.Points.FindObject("ENTITY 2 139 1")
        objects6[162] = point168
        point169 = workPart.Points.FindObject("ENTITY 2 138 1")
        objects6[163] = point169
        point170 = workPart.Points.FindObject("ENTITY 2 137 1")
        objects6[164] = point170
        point171 = workPart.Points.FindObject("ENTITY 2 136 1")
        objects6[165] = point171
        point172 = workPart.Points.FindObject("ENTITY 2 135 1")
        objects6[166] = point172
        point173 = workPart.Points.FindObject("ENTITY 2 134 1")
        objects6[167] = point173
        point174 = workPart.Points.FindObject("ENTITY 2 133 1")
        objects6[168] = point174
        point175 = workPart.Points.FindObject("ENTITY 2 132 1")
        objects6[169] = point175
        point176 = workPart.Points.FindObject("ENTITY 2 131 1")
        objects6[170] = point176
        point177 = workPart.Points.FindObject("ENTITY 2 130 1")
        objects6[171] = point177
        point178 = workPart.Points.FindObject("ENTITY 2 129 1")
        objects6[172] = point178
        point179 = workPart.Points.FindObject("ENTITY 2 128 1")
        objects6[173] = point179
        point180 = workPart.Points.FindObject("ENTITY 2 127 1")
        objects6[174] = point180
        point181 = workPart.Points.FindObject("ENTITY 2 126 1")
        objects6[175] = point181
        point182 = workPart.Points.FindObject("ENTITY 2 125 1")
        objects6[176] = point182
        point183 = workPart.Points.FindObject("ENTITY 2 124 1")
        objects6[177] = point183
        point184 = workPart.Points.FindObject("ENTITY 2 123 1")
        objects6[178] = point184
        point185 = workPart.Points.FindObject("ENTITY 2 122 1")
        objects6[179] = point185
        point186 = workPart.Points.FindObject("ENTITY 2 121 1")
        objects6[180] = point186
        point187 = workPart.Points.FindObject("ENTITY 2 120 1")
        objects6[181] = point187
        point188 = workPart.Points.FindObject("ENTITY 2 119 1")
        objects6[182] = point188
        point189 = workPart.Points.FindObject("ENTITY 2 118 1")
        objects6[183] = point189
        point190 = workPart.Points.FindObject("ENTITY 2 117 1")
        objects6[184] = point190
        point191 = workPart.Points.FindObject("ENTITY 2 116 1")
        objects6[185] = point191
        point192 = workPart.Points.FindObject("ENTITY 2 115 1")
        objects6[186] = point192
        point193 = workPart.Points.FindObject("ENTITY 2 114 1")
        objects6[187] = point193
        point194 = workPart.Points.FindObject("ENTITY 2 113 1")
        objects6[188] = point194
        point195 = workPart.Points.FindObject("ENTITY 2 112 1")
        objects6[189] = point195
        point196 = workPart.Points.FindObject("ENTITY 2 111 1")
        objects6[190] = point196
        point197 = workPart.Points.FindObject("ENTITY 2 110 1")
        objects6[191] = point197
        point198 = workPart.Points.FindObject("ENTITY 2 109 1")
        objects6[192] = point198
        point199 = workPart.Points.FindObject("ENTITY 2 108 1")
        objects6[193] = point199
        point200 = workPart.Points.FindObject("ENTITY 2 107 1")
        objects6[194] = point200
        point201 = workPart.Points.FindObject("ENTITY 2 106 1")
        objects6[195] = point201
        point202 = workPart.Points.FindObject("ENTITY 2 105 1")
        objects6[196] = point202
        point203 = workPart.Points.FindObject("ENTITY 2 104 1")
        objects6[197] = point203
        point204 = workPart.Points.FindObject("ENTITY 2 103 1")
        objects6[198] = point204
        point205 = workPart.Points.FindObject("ENTITY 2 102 1")
        objects6[199] = point205
        point206 = workPart.Points.FindObject("ENTITY 2 101 1")
        objects6[200] = point206
        point207 = workPart.Points.FindObject("ENTITY 2 100 1")
        objects6[201] = point207
        point208 = workPart.Points.FindObject("ENTITY 2 99 1")
        objects6[202] = point208
        point209 = workPart.Points.FindObject("ENTITY 2 98 1")
        objects6[203] = point209
        point210 = workPart.Points.FindObject("ENTITY 2 97 1")
        objects6[204] = point210
        point211 = workPart.Points.FindObject("ENTITY 2 96 1")
        objects6[205] = point211
        point212 = workPart.Points.FindObject("ENTITY 2 95 1")
        objects6[206] = point212
        point213 = workPart.Points.FindObject("ENTITY 2 94 1")
        objects6[207] = point213
        point214 = workPart.Points.FindObject("ENTITY 2 93 1")
        objects6[208] = point214
        point215 = workPart.Points.FindObject("ENTITY 2 92 1")
        objects6[209] = point215
        point216 = workPart.Points.FindObject("ENTITY 2 91 1")
        objects6[210] = point216
        point217 = workPart.Points.FindObject("ENTITY 2 90 1")
        objects6[211] = point217
        point218 = workPart.Points.FindObject("ENTITY 2 89 1")
        objects6[212] = point218
        point219 = workPart.Points.FindObject("ENTITY 2 88 1")
        objects6[213] = point219
        point220 = workPart.Points.FindObject("ENTITY 2 87 1")
        objects6[214] = point220
        point221 = workPart.Points.FindObject("ENTITY 2 86 1")
        objects6[215] = point221
        point222 = workPart.Points.FindObject("ENTITY 2 85 1")
        objects6[216] = point222
        point223 = workPart.Points.FindObject("ENTITY 2 84 1")
        objects6[217] = point223
        point224 = workPart.Points.FindObject("ENTITY 2 83 1")
        objects6[218] = point224
        point225 = workPart.Points.FindObject("ENTITY 2 82 1")
        objects6[219] = point225
        point226 = workPart.Points.FindObject("ENTITY 2 81 1")
        objects6[220] = point226
        point227 = workPart.Points.FindObject("ENTITY 2 80 1")
        objects6[221] = point227
        point228 = workPart.Points.FindObject("ENTITY 2 79 1")
        objects6[222] = point228
        point229 = workPart.Points.FindObject("ENTITY 2 78 1")
        objects6[223] = point229
        point230 = workPart.Points.FindObject("ENTITY 2 77 1")
        objects6[224] = point230
        point231 = workPart.Points.FindObject("ENTITY 2 76 1")
        objects6[225] = point231
        point232 = workPart.Points.FindObject("ENTITY 2 75 1")
        objects6[226] = point232
        point233 = workPart.Points.FindObject("ENTITY 2 74 1")
        objects6[227] = point233
        point234 = workPart.Points.FindObject("ENTITY 2 73 1")
        objects6[228] = point234
        point235 = workPart.Points.FindObject("ENTITY 2 72 1")
        objects6[229] = point235
        point236 = workPart.Points.FindObject("ENTITY 2 71 1")
        objects6[230] = point236
        point237 = workPart.Points.FindObject("ENTITY 2 70 1")
        objects6[231] = point237
        point238 = workPart.Points.FindObject("ENTITY 2 69 1")
        objects6[232] = point238
        point239 = workPart.Points.FindObject("ENTITY 2 68 1")
        objects6[233] = point239
        point240 = workPart.Points.FindObject("ENTITY 2 67 1")
        objects6[234] = point240
        point241 = workPart.Points.FindObject("ENTITY 2 66 1")
        objects6[235] = point241
        point242 = workPart.Points.FindObject("ENTITY 2 65 1")
        objects6[236] = point242
        point243 = workPart.Points.FindObject("ENTITY 2 64 1")
        objects6[237] = point243
        point244 = workPart.Points.FindObject("ENTITY 2 63 1")
        objects6[238] = point244
        point245 = workPart.Points.FindObject("ENTITY 2 62 1")
        objects6[239] = point245
        point246 = workPart.Points.FindObject("ENTITY 2 61 1")
        objects6[240] = point246
        point247 = workPart.Points.FindObject("ENTITY 2 60 1")
        objects6[241] = point247
        point248 = workPart.Points.FindObject("ENTITY 2 59 1")
        objects6[242] = point248
        point249 = workPart.Points.FindObject("ENTITY 2 58 1")
        objects6[243] = point249
        point250 = workPart.Points.FindObject("ENTITY 2 57 1")
        objects6[244] = point250
        point251 = workPart.Points.FindObject("ENTITY 2 56 1")
        objects6[245] = point251
        point252 = workPart.Points.FindObject("ENTITY 2 55 1")
        objects6[246] = point252
        point253 = workPart.Points.FindObject("ENTITY 2 54 1")
        objects6[247] = point253
        point254 = workPart.Points.FindObject("ENTITY 2 53 1")
        objects6[248] = point254
        point255 = workPart.Points.FindObject("ENTITY 2 52 1")
        objects6[249] = point255
        point256 = workPart.Points.FindObject("ENTITY 2 51 1")
        objects6[250] = point256
        point257 = workPart.Points.FindObject("ENTITY 2 50 1")
        objects6[251] = point257
        point258 = workPart.Points.FindObject("ENTITY 2 49 1")
        objects6[252] = point258
        point259 = workPart.Points.FindObject("ENTITY 2 48 1")
        objects6[253] = point259
        point260 = workPart.Points.FindObject("ENTITY 2 47 1")
        objects6[254] = point260
        point261 = workPart.Points.FindObject("ENTITY 2 46 1")
        objects6[255] = point261
        point262 = workPart.Points.FindObject("ENTITY 2 45 1")
        objects6[256] = point262
        point263 = workPart.Points.FindObject("ENTITY 2 44 1")
        objects6[257] = point263
        point264 = workPart.Points.FindObject("ENTITY 2 43 1")
        objects6[258] = point264
        point265 = workPart.Points.FindObject("ENTITY 2 42 1")
        objects6[259] = point265
        point266 = workPart.Points.FindObject("ENTITY 2 41 1")
        objects6[260] = point266
        point267 = workPart.Points.FindObject("ENTITY 2 40 1")
        objects6[261] = point267
        point268 = workPart.Points.FindObject("ENTITY 2 39 1")
        objects6[262] = point268
        point269 = workPart.Points.FindObject("ENTITY 2 38 1")
        objects6[263] = point269
        point270 = workPart.Points.FindObject("ENTITY 2 37 1")
        objects6[264] = point270
        point271 = workPart.Points.FindObject("ENTITY 2 36 1")
        objects6[265] = point271
        point272 = workPart.Points.FindObject("ENTITY 2 35 1")
        objects6[266] = point272
        point273 = workPart.Points.FindObject("ENTITY 2 34 1")
        objects6[267] = point273
        point274 = workPart.Points.FindObject("ENTITY 2 33 1")
        objects6[268] = point274
        point275 = workPart.Points.FindObject("ENTITY 2 32 1")
        objects6[269] = point275
        point276 = workPart.Points.FindObject("ENTITY 2 31 1")
        objects6[270] = point276
        point277 = workPart.Points.FindObject("ENTITY 2 30 1")
        objects6[271] = point277
        point278 = workPart.Points.FindObject("ENTITY 2 29 1")
        objects6[272] = point278
        point279 = workPart.Points.FindObject("ENTITY 2 28 1")
        objects6[273] = point279
        point280 = workPart.Points.FindObject("ENTITY 2 27 1")
        objects6[274] = point280
        point281 = workPart.Points.FindObject("ENTITY 2 26 1")
        objects6[275] = point281
        point282 = workPart.Points.FindObject("ENTITY 2 25 1")
        objects6[276] = point282
        point283 = workPart.Points.FindObject("ENTITY 2 24 1")
        objects6[277] = point283
        point284 = workPart.Points.FindObject("ENTITY 2 23 1")
        objects6[278] = point284
        point285 = workPart.Points.FindObject("ENTITY 2 22 1")
        objects6[279] = point285
        point286 = workPart.Points.FindObject("ENTITY 2 21 1")
        objects6[280] = point286
        point287 = workPart.Points.FindObject("ENTITY 2 20 1")
        objects6[281] = point287
        point288 = workPart.Points.FindObject("ENTITY 2 19 1")
        objects6[282] = point288
        point289 = workPart.Points.FindObject("ENTITY 2 18 1")
        objects6[283] = point289
        point290 = workPart.Points.FindObject("ENTITY 2 17 1")
        objects6[284] = point290
        point291 = workPart.Points.FindObject("ENTITY 2 16 1")
        objects6[285] = point291
        point292 = workPart.Points.FindObject("ENTITY 2 15 1")
        objects6[286] = point292
        point293 = workPart.Points.FindObject("ENTITY 2 14 1")
        objects6[287] = point293
        point294 = workPart.Points.FindObject("ENTITY 2 13 1")
        objects6[288] = point294
        point295 = workPart.Points.FindObject("ENTITY 2 12 1")
        objects6[289] = point295
        point296 = workPart.Points.FindObject("ENTITY 2 11 1")
        objects6[290] = point296
        point297 = workPart.Points.FindObject("ENTITY 2 10 1")
        objects6[291] = point297
        point298 = workPart.Points.FindObject("ENTITY 2 9 1")
        objects6[292] = point298
        point299 = workPart.Points.FindObject("ENTITY 2 8 1")
        objects6[293] = point299
        point300 = workPart.Points.FindObject("ENTITY 2 7 1")
        objects6[294] = point300
        point301 = workPart.Points.FindObject("ENTITY 2 6 1")
        objects6[295] = point301
        point302 = workPart.Points.FindObject("ENTITY 2 5 1")
        objects6[296] = point302
        point303 = workPart.Points.FindObject("ENTITY 2 4 1")
        objects6[297] = point303
        point304 = workPart.Points.FindObject("ENTITY 2 3 1")
        objects6[298] = point304
        point305 = workPart.Points.FindObject("ENTITY 2 2 1")
        objects6[299] = point305
        point306 = workPart.Points.FindObject("ENTITY 2 1 1")
        objects6[300] = point306
        group1 = nXObject5
        objects6[301] = group1
        added2 = fitCurveBuilder1.Target.Add(objects6)
        
        geometricConstraintData1 = fitCurveBuilder1.ConstraintManager.FindItem(0)
        
        point307 = geometricConstraintData1.Point
        
        geometricConstraintData2 = fitCurveBuilder1.ConstraintManager.FindItem(1)
        
        point308 = geometricConstraintData2.Point
        
        theSession.SetUndoMarkName(markId29, "拟合曲线 - 选择")
        
        theSession.SetUndoMarkVisibility(markId29, None, NXOpen.Session.MarkVisibility.Visible)
        
        theSession.SetUndoMarkVisibility(markId27, None, NXOpen.Session.MarkVisibility.Invisible)
        
        markId30 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        fitCurveBuilder1.HasReversedDirection = True
        
        theSession.SetUndoMarkName(markId30, "拟合曲线 - 反向")
        
        theSession.SetUndoMarkVisibility(markId30, None, NXOpen.Session.MarkVisibility.Visible)
        
        theSession.SetUndoMarkVisibility(markId27, None, NXOpen.Session.MarkVisibility.Invisible)
        
        markId31 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        theSession.DeleteUndoMark(markId31, None)
        
        markId32 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        markId33 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        theSession.DeleteUndoMark(markId33, None)
        
        markId34 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        nXObject6 = fitCurveBuilder1.Commit()
        
        theSession.DeleteUndoMark(markId34, None)
        
        theSession.SetUndoMarkName(markId27, "拟合曲线")
        
        fitCurveBuilder1.Destroy()
        
        theSession.SetUndoMarkVisibility(markId27, None, NXOpen.Session.MarkVisibility.Visible)
        
        theSession.DeleteUndoMark(markId30, None)
        
        theSession.DeleteUndoMark(markId29, None)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->5 抽取几何特征
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->关联复制(A)->抽取几何特征(E)...
        # ----------------------------------------------
        markId35 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        wavePointBuilder2 = workPart.Features.CreateWavePointBuilder(NXOpen.Features.Feature.Null)
        
        waveDatumBuilder2 = workPart.Features.CreateWaveDatumBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder2 = workPart.Features.CreateCompositeCurveBuilder(NXOpen.Features.Feature.Null)
        
        extractFaceBuilder2 = workPart.Features.CreateExtractFaceBuilder(NXOpen.Features.Feature.Null)
        
        mirrorBodyBuilder2 = workPart.Features.CreateMirrorBodyBuilder(NXOpen.Features.Feature.Null)
        
        waveSketchBuilder2 = workPart.Features.CreateWaveSketchBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder2.CurveFitData.Tolerance = 0.001
        
        compositeCurveBuilder2.CurveFitData.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder2.Section.SetAllowRefCrvs(False)
        
        extractFaceBuilder2.FaceOption = NXOpen.Features.ExtractFaceBuilder.FaceOptionType.AdjacentFaces
        
        compositeCurveBuilder2.Associative = False
        
        waveDatumBuilder2.ParentPart = NXOpen.Features.WaveDatumBuilder.ParentPartType.WorkPart
        
        wavePointBuilder2.ParentPart = NXOpen.Features.WavePointBuilder.ParentPartType.WorkPart
        
        extractFaceBuilder2.ParentPart = NXOpen.Features.ExtractFaceBuilder.ParentPartType.WorkPart
        
        mirrorBodyBuilder2.ParentPartType = NXOpen.Features.MirrorBodyBuilder.ParentPart.WorkPart
        
        compositeCurveBuilder2.ParentPart = NXOpen.Features.CompositeCurveBuilder.PartType.WorkPart
        
        waveSketchBuilder2.ParentPart = NXOpen.Features.WaveSketchBuilder.ParentPartType.WorkPart
        
        compositeCurveBuilder2.Associative = False
        
        theSession.SetUndoMarkName(markId35, "抽取几何特征 对话框")
        
        compositeCurveBuilder2.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder2.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder2.Section.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder2.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder2.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder2.Associative = False
        
        compositeCurveBuilder2.FixAtCurrentTimestamp = False
        
        compositeCurveBuilder2.HideOriginal = False
        
        compositeCurveBuilder2.InheritDisplayProperties = False
        
        compositeCurveBuilder2.Section.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.CurvesAndPoints)
        
        markId36 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId37 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions2 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions2.SetSelectedFromInactive(False)
        
        curves2 = [NXOpen.IBaseCurve.Null] * 1 
        spline1 = workPart.Splines.FindObject("ENTITY 9 1 1")
        curves2[0] = spline1
        curveDumbRule2 = workPart.ScRuleFactory.CreateRuleBaseCurveDumb(curves2, selectionIntentRuleOptions2)
        
        selectionIntentRuleOptions2.Dispose()
        compositeCurveBuilder2.Section.AllowSelfIntersection(False)
        
        compositeCurveBuilder2.Section.AllowDegenerateCurves(False)
        
        rules2 = [None] * 1 
        rules2[0] = curveDumbRule2
        helpPoint2 = NXOpen.Point3d(9.2722845133269551, 0.071273375427821861, -0.027196341045371164)
        compositeCurveBuilder2.Section.AddToSection(rules2, spline1, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint2, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId37, None)
        
        theSession.DeleteUndoMark(markId36, None)
        
        markId38 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        theSession.DeleteUndoMark(markId38, None)
        
        markId39 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        nXObject7 = compositeCurveBuilder2.Commit()
        
        theSession.DeleteUndoMark(markId39, None)
        
        theSession.SetUndoMarkName(markId35, "抽取几何特征")
        
        compositeCurveBuilder2.Destroy()
        
        waveDatumBuilder2.Destroy()
        
        wavePointBuilder2.Destroy()
        
        extractFaceBuilder2.Destroy()
        
        mirrorBodyBuilder2.Destroy()
        
        waveSketchBuilder2.Destroy()
        
        markId40 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects7 = [NXOpen.DisplayableObject.Null] * 1 
        compositeCurve2 = nXObject7
        spline2 = compositeCurve2.FindObject("CURVE 1")
        objects7[0] = spline2
        theSession.DisplayManager.BlankObjects(objects7)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：编辑(E)->选择(L)->全选(S)
        # ----------------------------------------------
        # Refer to the sample NXOpen application, Selection for "Select All" alternatives.
        # ----------------------------------------------
        #   菜单：编辑(E)->删除(D)...
        # ----------------------------------------------
        markId41 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Delete")
        
        theSession.UpdateManager.ClearErrorList()
        
        markId42 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Delete")
        
        objects8 = [NXOpen.TaggedObject.Null] * 303 
        objects8[0] = point6
        objects8[1] = point7
        objects8[2] = point8
        objects8[3] = point9
        objects8[4] = point10
        objects8[5] = point11
        objects8[6] = point12
        objects8[7] = point13
        objects8[8] = point14
        objects8[9] = point15
        objects8[10] = point16
        objects8[11] = point17
        objects8[12] = point18
        objects8[13] = point19
        objects8[14] = point20
        objects8[15] = point21
        objects8[16] = point22
        objects8[17] = point23
        objects8[18] = point24
        objects8[19] = point25
        objects8[20] = point26
        objects8[21] = point27
        objects8[22] = point28
        objects8[23] = point29
        objects8[24] = point30
        objects8[25] = point31
        objects8[26] = point32
        objects8[27] = point33
        objects8[28] = point34
        objects8[29] = point35
        objects8[30] = point36
        objects8[31] = point37
        objects8[32] = point38
        objects8[33] = point39
        objects8[34] = point40
        objects8[35] = point41
        objects8[36] = point42
        objects8[37] = point43
        objects8[38] = point44
        objects8[39] = point45
        objects8[40] = point270
        objects8[41] = point271
        objects8[42] = point272
        objects8[43] = point273
        objects8[44] = point274
        objects8[45] = point275
        objects8[46] = point276
        objects8[47] = point277
        objects8[48] = point278
        objects8[49] = point279
        objects8[50] = point280
        objects8[51] = point281
        objects8[52] = point282
        objects8[53] = point283
        objects8[54] = point284
        objects8[55] = point285
        objects8[56] = point286
        objects8[57] = point287
        objects8[58] = point288
        objects8[59] = point289
        objects8[60] = point290
        objects8[61] = point291
        objects8[62] = point292
        objects8[63] = point293
        objects8[64] = point294
        objects8[65] = point295
        objects8[66] = point296
        objects8[67] = point297
        objects8[68] = point298
        objects8[69] = point299
        objects8[70] = point300
        objects8[71] = point301
        objects8[72] = point302
        objects8[73] = point303
        objects8[74] = point304
        objects8[75] = point305
        objects8[76] = point306
        objects8[77] = group1
        objects8[78] = point161
        objects8[79] = point162
        objects8[80] = point163
        objects8[81] = point164
        objects8[82] = point165
        objects8[83] = point166
        objects8[84] = point167
        objects8[85] = point168
        objects8[86] = point169
        objects8[87] = point170
        objects8[88] = point171
        objects8[89] = point172
        objects8[90] = point173
        objects8[91] = point174
        objects8[92] = point175
        objects8[93] = point176
        objects8[94] = point177
        objects8[95] = point178
        objects8[96] = point179
        objects8[97] = point180
        objects8[98] = point181
        objects8[99] = point182
        objects8[100] = point183
        objects8[101] = point184
        objects8[102] = point185
        objects8[103] = point186
        objects8[104] = point187
        objects8[105] = point188
        objects8[106] = point189
        objects8[107] = point190
        objects8[108] = point191
        objects8[109] = point192
        objects8[110] = point193
        objects8[111] = point194
        objects8[112] = point195
        objects8[113] = point196
        objects8[114] = point197
        objects8[115] = point198
        objects8[116] = point199
        objects8[117] = point200
        objects8[118] = point201
        objects8[119] = point202
        objects8[120] = point203
        objects8[121] = point204
        objects8[122] = point205
        objects8[123] = point206
        objects8[124] = point73
        objects8[125] = point74
        objects8[126] = point75
        objects8[127] = point76
        objects8[128] = point77
        objects8[129] = point78
        objects8[130] = point79
        objects8[131] = point80
        objects8[132] = point81
        objects8[133] = point82
        objects8[134] = point83
        objects8[135] = point84
        objects8[136] = point85
        objects8[137] = point86
        objects8[138] = point87
        objects8[139] = point88
        objects8[140] = point89
        objects8[141] = point90
        objects8[142] = point91
        objects8[143] = point92
        objects8[144] = point93
        objects8[145] = point94
        objects8[146] = point95
        objects8[147] = point96
        objects8[148] = point97
        objects8[149] = point98
        objects8[150] = point99
        objects8[151] = point100
        objects8[152] = point101
        objects8[153] = point102
        objects8[154] = point103
        objects8[155] = point104
        objects8[156] = point105
        objects8[157] = point106
        objects8[158] = point107
        objects8[159] = point108
        objects8[160] = point109
        objects8[161] = point110
        objects8[162] = point111
        objects8[163] = point112
        objects8[164] = point113
        objects8[165] = point114
        objects8[166] = point115
        objects8[167] = point116
        objects8[168] = point117
        objects8[169] = point118
        objects8[170] = point119
        objects8[171] = point120
        objects8[172] = point121
        objects8[173] = point122
        objects8[174] = point123
        objects8[175] = point124
        objects8[176] = point125
        objects8[177] = point46
        objects8[178] = point47
        objects8[179] = point48
        objects8[180] = point49
        objects8[181] = point50
        objects8[182] = point51
        objects8[183] = point52
        objects8[184] = point53
        objects8[185] = point54
        objects8[186] = point55
        objects8[187] = spline1
        objects8[188] = point207
        objects8[189] = point208
        objects8[190] = point209
        objects8[191] = point210
        objects8[192] = point211
        objects8[193] = point212
        objects8[194] = point213
        objects8[195] = point214
        objects8[196] = point215
        objects8[197] = point216
        objects8[198] = point217
        objects8[199] = point218
        objects8[200] = point219
        objects8[201] = point220
        objects8[202] = point221
        objects8[203] = point222
        objects8[204] = point223
        objects8[205] = point224
        objects8[206] = point225
        objects8[207] = point226
        objects8[208] = point227
        objects8[209] = point228
        objects8[210] = point229
        objects8[211] = point230
        objects8[212] = point231
        objects8[213] = point232
        objects8[214] = point233
        objects8[215] = point234
        objects8[216] = point235
        objects8[217] = point236
        objects8[218] = point237
        objects8[219] = point238
        objects8[220] = point239
        objects8[221] = point240
        objects8[222] = point241
        objects8[223] = point242
        objects8[224] = point243
        objects8[225] = point244
        objects8[226] = point245
        objects8[227] = point246
        objects8[228] = point247
        objects8[229] = point248
        objects8[230] = point249
        objects8[231] = point250
        objects8[232] = point251
        objects8[233] = point252
        objects8[234] = point253
        objects8[235] = point254
        objects8[236] = point255
        objects8[237] = point256
        objects8[238] = point257
        objects8[239] = point258
        objects8[240] = point259
        objects8[241] = point260
        objects8[242] = point261
        objects8[243] = point262
        objects8[244] = point263
        objects8[245] = point264
        objects8[246] = point265
        objects8[247] = point266
        objects8[248] = point267
        objects8[249] = point268
        objects8[250] = point269
        objects8[251] = point126
        objects8[252] = point127
        objects8[253] = point128
        objects8[254] = point129
        objects8[255] = point130
        objects8[256] = point131
        objects8[257] = point132
        objects8[258] = point133
        objects8[259] = point134
        objects8[260] = point135
        objects8[261] = point136
        objects8[262] = point137
        objects8[263] = point138
        objects8[264] = point139
        objects8[265] = point140
        objects8[266] = point141
        objects8[267] = point142
        objects8[268] = point143
        objects8[269] = point144
        objects8[270] = point145
        objects8[271] = point146
        objects8[272] = point147
        objects8[273] = point148
        objects8[274] = point149
        objects8[275] = point150
        objects8[276] = point151
        objects8[277] = point152
        objects8[278] = point153
        objects8[279] = point154
        objects8[280] = point155
        objects8[281] = point156
        objects8[282] = point157
        objects8[283] = point158
        objects8[284] = point159
        objects8[285] = point160
        objects8[286] = point56
        objects8[287] = point57
        objects8[288] = point58
        objects8[289] = point59
        objects8[290] = point60
        objects8[291] = point61
        objects8[292] = point62
        objects8[293] = point63
        objects8[294] = point64
        objects8[295] = point65
        objects8[296] = point66
        objects8[297] = point67
        objects8[298] = point68
        objects8[299] = point69
        objects8[300] = point70
        objects8[301] = point71
        objects8[302] = point72
        nErrs5 = theSession.UpdateManager.AddObjectsToDeleteList(objects8)
        
        id2 = theSession.NewestVisibleUndoMark
        
        nErrs6 = theSession.UpdateManager.DoUpdate(id2)
        
        theSession.DeleteUndoMark(markId41, None)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->10 扫掠
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->扫掠(W)->扫掠(S)...
        # ----------------------------------------------
        markId43 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        sweptBuilder1 = workPart.Features.CreateSweptBuilder(NXOpen.Features.Swept.Null)
        
        expression28 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        expression29 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        expression30 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        sweptBuilder1.G0Tolerance = 0.001
        
        sweptBuilder1.G1Tolerance = 0.050000000000000003
        
        sweptBuilder1.OrientationMethod.AngularLaw.Value.SetFormula("0")
        
        sweptBuilder1.OrientationMethod.AngularLaw.StartValue.SetFormula("0")
        
        sweptBuilder1.OrientationMethod.AngularLaw.EndValue.SetFormula("0")
        
        sweptBuilder1.ScalingMethod.AreaLaw.Value.SetFormula("1")
        
        sweptBuilder1.ScalingMethod.AreaLaw.StartValue.SetFormula("1")
        
        sweptBuilder1.ScalingMethod.AreaLaw.EndValue.SetFormula("1")
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.Value.SetFormula("1")
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.StartValue.SetFormula("1")
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.EndValue.SetFormula("1")
        
        theSession.SetUndoMarkName(markId43, "扫掠 对话框")
        
        sweptBuilder1.Spine.DistanceTolerance = 0.001
        
        sweptBuilder1.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder1.AlignmentMethod.AlignCurve.DistanceTolerance = 0.001
        
        sweptBuilder1.AlignmentMethod.AlignCurve.ChainingTolerance = 0.00095
        
        sweptBuilder1.OrientationMethod.OrientationCurve.DistanceTolerance = 0.001
        
        sweptBuilder1.OrientationMethod.OrientationCurve.ChainingTolerance = 0.00095
        
        sweptBuilder1.OrientationMethod.AngularLaw.AlongSpineData.Spine.DistanceTolerance = 0.001
        
        sweptBuilder1.OrientationMethod.AngularLaw.AlongSpineData.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder1.OrientationMethod.AngularLaw.LawCurve.DistanceTolerance = 0.001
        
        sweptBuilder1.OrientationMethod.AngularLaw.LawCurve.ChainingTolerance = 0.00095
        
        sweptBuilder1.ScalingMethod.ScalingCurve.DistanceTolerance = 0.001
        
        sweptBuilder1.ScalingMethod.ScalingCurve.ChainingTolerance = 0.00095
        
        sweptBuilder1.ScalingMethod.AreaLaw.AlongSpineData.Spine.DistanceTolerance = 0.001
        
        sweptBuilder1.ScalingMethod.AreaLaw.AlongSpineData.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder1.ScalingMethod.AreaLaw.LawCurve.DistanceTolerance = 0.001
        
        sweptBuilder1.ScalingMethod.AreaLaw.LawCurve.ChainingTolerance = 0.00095
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.AlongSpineData.Spine.DistanceTolerance = 0.001
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.AlongSpineData.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.LawCurve.DistanceTolerance = 0.001
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.LawCurve.ChainingTolerance = 0.00095
        
        sweptBuilder1.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.AlignmentMethod.AlignCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.OrientationMethod.OrientationCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.OrientationMethod.AngularLaw.AlongSpineData.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.OrientationMethod.AngularLaw.LawCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.ScalingMethod.ScalingCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.ScalingMethod.AreaLaw.AlongSpineData.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.ScalingMethod.AreaLaw.LawCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.AlongSpineData.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.LawCurve.AngleTolerance = 0.050000000000000003
        
        section1 = workPart.Sections.CreateSection(0.00095, 0.001, 0.050000000000000003)
        
        sweptBuilder1.SectionList.Append(section1)
        
        section1.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.OnlyCurves)
        
        markId44 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId45 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions3 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions3.SetSelectedFromInactive(False)
        
        features1 = [NXOpen.Features.Feature.Null] * 1 
        features1[0] = compositeCurve1
        curveFeatureRule1 = workPart.ScRuleFactory.CreateRuleCurveFeature(features1, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions3)
        
        selectionIntentRuleOptions3.Dispose()
        section1.AllowSelfIntersection(False)
        
        section1.AllowDegenerateCurves(False)
        
        rules3 = [None] * 1 
        rules3[0] = curveFeatureRule1
        helpPoint3 = NXOpen.Point3d(0.0, 0.0, 0.0)
        section1.AddToSection(rules3, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint3, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId45, None)
        
        sections1 = [NXOpen.Section.Null] * 1 
        sections1[0] = section1
        sweptBuilder1.AlignmentMethod.SetSections(sections1)
        
        expression31 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        theSession.DeleteUndoMark(markId44, None)
        
        section1.ReverseDirectionOfLoop(0)
        
        try:
            sweptBuilder1.AlignmentMethod.UpdateSectionAtIndex(0)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1)
            
        section2 = workPart.Sections.CreateSection(0.00095, 0.001, 0.050000000000000003)
        
        sweptBuilder1.GuideList.Append(section2)
        
        section2.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.OnlyCurves)
        
        markId46 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId47 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions4 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions4.SetSelectedFromInactive(False)
        
        features2 = [NXOpen.Features.Feature.Null] * 1 
        features2[0] = compositeCurve2
        curveFeatureRule2 = workPart.ScRuleFactory.CreateRuleCurveFeature(features2, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions4)
        
        selectionIntentRuleOptions4.Dispose()
        section2.AllowSelfIntersection(False)
        
        section2.AllowDegenerateCurves(False)
        
        rules4 = [None] * 1 
        rules4[0] = curveFeatureRule2
        helpPoint4 = NXOpen.Point3d(0.0, 0.0, 0.0)
        section2.AddToSection(rules4, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint4, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId47, None)
        
        sweptBuilder1.ScalingMethod.AreaLaw.AlongSpineData.SetFeatureSpine(section2)
        
        sweptBuilder1.ScalingMethod.PerimeterLaw.AlongSpineData.SetFeatureSpine(section2)
        
        markId48 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId48, "Update Law Data", False)
        
        markId49 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId49, "Update Law Data", False)
        
        sweptBuilder1.OrientationMethod.AngularLaw.AlongSpineData.SetFeatureSpine(section2)
        
        markId50 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId50, "Update Law Data", False)
        
        theSession.DeleteUndoMark(markId46, None)
        
        markId51 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "扫掠")
        
        theSession.DeleteUndoMark(markId51, None)
        
        markId52 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "扫掠")
        
        nXObject8 = sweptBuilder1.Commit()
        
        displayModification1 = theSession.DisplayManager.NewDisplayModification()
        
        displayModification1.ApplyToAllFaces = False
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects9 = [NXOpen.DisplayableObject.Null] * 1 
        swept1 = nXObject8
        face1 = swept1.FindObject("FACE 10001 {(256.5918744660966,4.6033091000688,0.9305407347416) SWEPT(3)}")
        objects9[0] = face1
        displayModification1.Apply(objects9)
        
        face1.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects10 = [NXOpen.DisplayableObject.Null] * 1 
        face2 = swept1.FindObject("FACE 1 {(0,0.9062566580345,3.6078924041855) SWEPT(3)}")
        objects10[0] = face2
        displayModification1.Apply(objects10)
        
        face2.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects11 = [NXOpen.DisplayableObject.Null] * 1 
        face3 = swept1.FindObject("FACE 10002 {(256.3040963940989,7.9200275666917,1.7824586350712) SWEPT(3)}")
        objects11[0] = face3
        displayModification1.Apply(objects11)
        
        face3.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects12 = [NXOpen.DisplayableObject.Null] * 1 
        face4 = swept1.FindObject("FACE 2 {(508.4494855861934,52.2833851684264,-11.8255087932661) SWEPT(3)}")
        objects12[0] = face4
        displayModification1.Apply(objects12)
        
        face4.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects13 = [NXOpen.DisplayableObject.Null] * 1 
        face5 = swept1.FindObject("FACE 10020 {(256.7035974451371,3.4693723671369,1.1088943948628) SWEPT(3)}")
        objects13[0] = face5
        displayModification1.Apply(objects13)
        
        face5.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects14 = [NXOpen.DisplayableObject.Null] * 1 
        face6 = swept1.FindObject("FACE 10003 {(255.9742708196916,11.4603860188839,1.8944626876613) SWEPT(3)}")
        objects14[0] = face6
        displayModification1.Apply(objects14)
        
        face6.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects15 = [NXOpen.DisplayableObject.Null] * 1 
        face7 = swept1.FindObject("FACE 10004 {(255.8800222862216,11.7368466981245,-0.5087043990157) SWEPT(3)}")
        objects15[0] = face7
        displayModification1.Apply(objects15)
        
        face7.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects16 = [NXOpen.DisplayableObject.Null] * 1 
        face8 = swept1.FindObject("FACE 10005 {(255.7166357302052,12.5675944390167,-3.5105716784988) SWEPT(3)}")
        objects16[0] = face8
        displayModification1.Apply(objects16)
        
        face8.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects17 = [NXOpen.DisplayableObject.Null] * 1 
        face9 = swept1.FindObject("FACE 10006 {(255.6079294371676,14.031966256208,-2.4882184234099) SWEPT(3)}")
        objects17[0] = face9
        displayModification1.Apply(objects17)
        
        face9.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects18 = [NXOpen.DisplayableObject.Null] * 1 
        face10 = swept1.FindObject("FACE 10007 {(255.7071053895501,12.9531447451625,-2.5691415293955) SWEPT(3)}")
        objects18[0] = face10
        displayModification1.Apply(objects18)
        
        face10.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects19 = [NXOpen.DisplayableObject.Null] * 1 
        face11 = swept1.FindObject("FACE 10008 {(255.7847887221576,12.7497368353658,-0.507333931295) SWEPT(3)}")
        objects19[0] = face11
        displayModification1.Apply(objects19)
        
        face11.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects20 = [NXOpen.DisplayableObject.Null] * 1 
        face12 = swept1.FindObject("FACE 10009 {(255.8264589308305,13.0324914616011,1.8965897887985) SWEPT(3)}")
        objects20[0] = face12
        displayModification1.Apply(objects20)
        
        face12.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects21 = [NXOpen.DisplayableObject.Null] * 1 
        face13 = swept1.FindObject("FACE 10010 {(255.4905704804347,16.5725691620772,1.7941657576135) SWEPT(3)}")
        objects21[0] = face13
        displayModification1.Apply(objects21)
        
        face13.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects22 = [NXOpen.DisplayableObject.Null] * 1 
        face14 = swept1.FindObject("FACE 10011 {(255.06783386596,20.5470559202536,0.0723328185484) SWEPT(3)}")
        objects22[0] = face14
        displayModification1.Apply(objects22)
        
        face14.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects23 = [NXOpen.DisplayableObject.Null] * 1 
        face15 = swept1.FindObject("FACE 10012 {(255.1064254440002,19.357618565584,-2.5083942060097) SWEPT(3)}")
        objects23[0] = face15
        displayModification1.Apply(objects23)
        
        face15.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects24 = [NXOpen.DisplayableObject.Null] * 1 
        face16 = swept1.FindObject("FACE 10013 {(255.3184523806084,17.1204077090706,-2.4522347668182) SWEPT(3)}")
        objects24[0] = face16
        displayModification1.Apply(objects24)
        
        face16.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects25 = [NXOpen.DisplayableObject.Null] * 1 
        face17 = swept1.FindObject("FACE 10014 {(255.0519208200072,19.6426964088287,-3.4834768719117) SWEPT(3)}")
        objects25[0] = face17
        displayModification1.Apply(objects25)
        
        face17.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects26 = [NXOpen.DisplayableObject.Null] * 1 
        face18 = swept1.FindObject("FACE 10015 {(254.9925413530571,21.4728406962011,0.487394568034) SWEPT(3)}")
        objects26[0] = face18
        displayModification1.Apply(objects26)
        
        face18.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects27 = [NXOpen.DisplayableObject.Null] * 1 
        face19 = swept1.FindObject("FACE 10016 {(255.5018041804894,16.7552788933296,2.7949261953241) SWEPT(3)}")
        objects27[0] = face19
        displayModification1.Apply(objects27)
        
        face19.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects28 = [NXOpen.DisplayableObject.Null] * 1 
        face20 = swept1.FindObject("FACE 10017 {(255.9371746736661,12.2481432801491,3.1973926381291) SWEPT(3)}")
        objects28[0] = face20
        displayModification1.Apply(objects28)
        
        face20.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects29 = [NXOpen.DisplayableObject.Null] * 1 
        face21 = swept1.FindObject("FACE 10018 {(256.3494410885942,7.7399377935902,2.7827281952507) SWEPT(3)}")
        objects29[0] = face21
        displayModification1.Apply(objects29)
        
        face21.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects30 = [NXOpen.DisplayableObject.Null] * 1 
        face22 = swept1.FindObject("FACE 10019 {(256.6474963467662,4.3051505053403,1.9016400175526) SWEPT(3)}")
        objects30[0] = face22
        displayModification1.Apply(objects30)
        
        face22.Color = 32767
        
        theSession.DeleteUndoMark(markId52, None)
        
        theSession.SetUndoMarkName(markId43, "扫掠")
        
        sweptBuilder1.Destroy()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression31)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression28)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression29)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression30)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        markId53 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects31 = [NXOpen.DisplayableObject.Null] * 1 
        body1 = workPart.Bodies.FindObject("SWEPT(3)")
        objects31[0] = body1
        theSession.DisplayManager.BlankObjects(objects31)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：文件(F)->导入(M)->文件中的点(L)...
        # ----------------------------------------------
        markId54 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Import Points from File")
        
        pointsFromFileBuilder2 = workPart.CreatePointsFromFileBuilder()
        
        pointsFromFileBuilder2.FileName = prediction_line_path
        
        pointsFromFileBuilder2.CoordinateOption = NXOpen.GeometricUtilities.PointsFromFileBuilder.Options.Absolute
        
        nXObject9 = pointsFromFileBuilder2.Commit()
        
        pointsFromFileBuilder2.Destroy()
        
        nErrs7 = theSession.UpdateManager.DoUpdate(markId54)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->5 拟合曲线
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->曲线(C)->拟合曲线(F)...
        # ----------------------------------------------
        markId55 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        fitCurveBuilder2 = workPart.Features.CreateFitCurveBuilder(NXOpen.Features.FitCurve.Null)
        
        fitCurveBuilder2.Tolerance = 0.001
        
        fitCurveBuilder2.TargetSourceType = NXOpen.Features.FitCurveBuilder.TargetSourceTypes.SpecifiedPoints
        
        fitCurveBuilder2.ProjectionDirectionOption = NXOpen.Features.FitCurveBuilder.ProjectionDirectionOptions.Normal
        
        fitCurveBuilder2.Radius.SetFormula("50")
        
        fitCurveBuilder2.Degree = 24
        
        fitCurveBuilder2.HasUniformSegments = True
        
        fitCurveBuilder2.Extender.StartValue.SetFormula("0")
        
        fitCurveBuilder2.Extender.EndValue.SetFormula("0")
        
        fitCurveBuilder2.RejectionThreshold.SetFormula("10")
        
        fitCurveBuilder2.IsAssociative = False
        
        theSession.SetUndoMarkName(markId55, "拟合曲线 对话框")
        
        markId56 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        theSession.DeleteUndoMark(markId56, None)
        
        markId57 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        # ----------------------------------------------
        #   菜单：编辑(E)->选择(L)->全选(S)
        # ----------------------------------------------
        # Refer to the sample NXOpen application, Selection for "Select All" alternatives.
        objects32 = [NXOpen.TaggedObject.Null] * 302 
        point309 = workPart.Points.FindObject("ENTITY 2 301 1")
        objects32[0] = point309
        point310 = workPart.Points.FindObject("ENTITY 2 300 1")
        objects32[1] = point310
        point311 = workPart.Points.FindObject("ENTITY 2 299 1")
        objects32[2] = point311
        point312 = workPart.Points.FindObject("ENTITY 2 298 1")
        objects32[3] = point312
        point313 = workPart.Points.FindObject("ENTITY 2 297 1")
        objects32[4] = point313
        point314 = workPart.Points.FindObject("ENTITY 2 296 1")
        objects32[5] = point314
        point315 = workPart.Points.FindObject("ENTITY 2 295 1")
        objects32[6] = point315
        point316 = workPart.Points.FindObject("ENTITY 2 294 1")
        objects32[7] = point316
        point317 = workPart.Points.FindObject("ENTITY 2 293 1")
        objects32[8] = point317
        point318 = workPart.Points.FindObject("ENTITY 2 292 1")
        objects32[9] = point318
        point319 = workPart.Points.FindObject("ENTITY 2 291 1")
        objects32[10] = point319
        point320 = workPart.Points.FindObject("ENTITY 2 290 1")
        objects32[11] = point320
        point321 = workPart.Points.FindObject("ENTITY 2 289 1")
        objects32[12] = point321
        point322 = workPart.Points.FindObject("ENTITY 2 288 1")
        objects32[13] = point322
        point323 = workPart.Points.FindObject("ENTITY 2 287 1")
        objects32[14] = point323
        point324 = workPart.Points.FindObject("ENTITY 2 286 1")
        objects32[15] = point324
        point325 = workPart.Points.FindObject("ENTITY 2 285 1")
        objects32[16] = point325
        point326 = workPart.Points.FindObject("ENTITY 2 284 1")
        objects32[17] = point326
        point327 = workPart.Points.FindObject("ENTITY 2 283 1")
        objects32[18] = point327
        point328 = workPart.Points.FindObject("ENTITY 2 282 1")
        objects32[19] = point328
        point329 = workPart.Points.FindObject("ENTITY 2 281 1")
        objects32[20] = point329
        point330 = workPart.Points.FindObject("ENTITY 2 280 1")
        objects32[21] = point330
        point331 = workPart.Points.FindObject("ENTITY 2 279 1")
        objects32[22] = point331
        point332 = workPart.Points.FindObject("ENTITY 2 278 1")
        objects32[23] = point332
        point333 = workPart.Points.FindObject("ENTITY 2 277 1")
        objects32[24] = point333
        point334 = workPart.Points.FindObject("ENTITY 2 276 1")
        objects32[25] = point334
        point335 = workPart.Points.FindObject("ENTITY 2 275 1")
        objects32[26] = point335
        point336 = workPart.Points.FindObject("ENTITY 2 274 1")
        objects32[27] = point336
        point337 = workPart.Points.FindObject("ENTITY 2 273 1")
        objects32[28] = point337
        point338 = workPart.Points.FindObject("ENTITY 2 272 1")
        objects32[29] = point338
        point339 = workPart.Points.FindObject("ENTITY 2 271 1")
        objects32[30] = point339
        point340 = workPart.Points.FindObject("ENTITY 2 270 1")
        objects32[31] = point340
        point341 = workPart.Points.FindObject("ENTITY 2 269 1")
        objects32[32] = point341
        point342 = workPart.Points.FindObject("ENTITY 2 268 1")
        objects32[33] = point342
        point343 = workPart.Points.FindObject("ENTITY 2 267 1")
        objects32[34] = point343
        point344 = workPart.Points.FindObject("ENTITY 2 266 1")
        objects32[35] = point344
        point345 = workPart.Points.FindObject("ENTITY 2 265 1")
        objects32[36] = point345
        point346 = workPart.Points.FindObject("ENTITY 2 264 1")
        objects32[37] = point346
        point347 = workPart.Points.FindObject("ENTITY 2 263 1")
        objects32[38] = point347
        point348 = workPart.Points.FindObject("ENTITY 2 262 1")
        objects32[39] = point348
        point349 = workPart.Points.FindObject("ENTITY 2 261 1")
        objects32[40] = point349
        point350 = workPart.Points.FindObject("ENTITY 2 260 1")
        objects32[41] = point350
        point351 = workPart.Points.FindObject("ENTITY 2 259 1")
        objects32[42] = point351
        point352 = workPart.Points.FindObject("ENTITY 2 258 1")
        objects32[43] = point352
        point353 = workPart.Points.FindObject("ENTITY 2 257 1")
        objects32[44] = point353
        point354 = workPart.Points.FindObject("ENTITY 2 256 1")
        objects32[45] = point354
        point355 = workPart.Points.FindObject("ENTITY 2 255 1")
        objects32[46] = point355
        point356 = workPart.Points.FindObject("ENTITY 2 254 1")
        objects32[47] = point356
        point357 = workPart.Points.FindObject("ENTITY 2 253 1")
        objects32[48] = point357
        point358 = workPart.Points.FindObject("ENTITY 2 252 1")
        objects32[49] = point358
        point359 = workPart.Points.FindObject("ENTITY 2 251 1")
        objects32[50] = point359
        point360 = workPart.Points.FindObject("ENTITY 2 250 1")
        objects32[51] = point360
        point361 = workPart.Points.FindObject("ENTITY 2 249 1")
        objects32[52] = point361
        point362 = workPart.Points.FindObject("ENTITY 2 248 1")
        objects32[53] = point362
        point363 = workPart.Points.FindObject("ENTITY 2 247 1")
        objects32[54] = point363
        point364 = workPart.Points.FindObject("ENTITY 2 246 1")
        objects32[55] = point364
        point365 = workPart.Points.FindObject("ENTITY 2 245 1")
        objects32[56] = point365
        point366 = workPart.Points.FindObject("ENTITY 2 244 1")
        objects32[57] = point366
        point367 = workPart.Points.FindObject("ENTITY 2 243 1")
        objects32[58] = point367
        point368 = workPart.Points.FindObject("ENTITY 2 242 1")
        objects32[59] = point368
        point369 = workPart.Points.FindObject("ENTITY 2 241 1")
        objects32[60] = point369
        point370 = workPart.Points.FindObject("ENTITY 2 240 1")
        objects32[61] = point370
        point371 = workPart.Points.FindObject("ENTITY 2 239 1")
        objects32[62] = point371
        point372 = workPart.Points.FindObject("ENTITY 2 238 1")
        objects32[63] = point372
        point373 = workPart.Points.FindObject("ENTITY 2 237 1")
        objects32[64] = point373
        point374 = workPart.Points.FindObject("ENTITY 2 236 1")
        objects32[65] = point374
        point375 = workPart.Points.FindObject("ENTITY 2 235 1")
        objects32[66] = point375
        point376 = workPart.Points.FindObject("ENTITY 2 234 1")
        objects32[67] = point376
        point377 = workPart.Points.FindObject("ENTITY 2 233 1")
        objects32[68] = point377
        point378 = workPart.Points.FindObject("ENTITY 2 232 1")
        objects32[69] = point378
        point379 = workPart.Points.FindObject("ENTITY 2 231 1")
        objects32[70] = point379
        point380 = workPart.Points.FindObject("ENTITY 2 230 1")
        objects32[71] = point380
        point381 = workPart.Points.FindObject("ENTITY 2 229 1")
        objects32[72] = point381
        point382 = workPart.Points.FindObject("ENTITY 2 228 1")
        objects32[73] = point382
        point383 = workPart.Points.FindObject("ENTITY 2 227 1")
        objects32[74] = point383
        point384 = workPart.Points.FindObject("ENTITY 2 226 1")
        objects32[75] = point384
        point385 = workPart.Points.FindObject("ENTITY 2 225 1")
        objects32[76] = point385
        point386 = workPart.Points.FindObject("ENTITY 2 224 1")
        objects32[77] = point386
        point387 = workPart.Points.FindObject("ENTITY 2 223 1")
        objects32[78] = point387
        point388 = workPart.Points.FindObject("ENTITY 2 222 1")
        objects32[79] = point388
        point389 = workPart.Points.FindObject("ENTITY 2 221 1")
        objects32[80] = point389
        point390 = workPart.Points.FindObject("ENTITY 2 220 1")
        objects32[81] = point390
        point391 = workPart.Points.FindObject("ENTITY 2 219 1")
        objects32[82] = point391
        point392 = workPart.Points.FindObject("ENTITY 2 218 1")
        objects32[83] = point392
        point393 = workPart.Points.FindObject("ENTITY 2 217 1")
        objects32[84] = point393
        point394 = workPart.Points.FindObject("ENTITY 2 216 1")
        objects32[85] = point394
        point395 = workPart.Points.FindObject("ENTITY 2 215 1")
        objects32[86] = point395
        point396 = workPart.Points.FindObject("ENTITY 2 214 1")
        objects32[87] = point396
        point397 = workPart.Points.FindObject("ENTITY 2 213 1")
        objects32[88] = point397
        point398 = workPart.Points.FindObject("ENTITY 2 212 1")
        objects32[89] = point398
        point399 = workPart.Points.FindObject("ENTITY 2 211 1")
        objects32[90] = point399
        point400 = workPart.Points.FindObject("ENTITY 2 210 1")
        objects32[91] = point400
        point401 = workPart.Points.FindObject("ENTITY 2 209 1")
        objects32[92] = point401
        point402 = workPart.Points.FindObject("ENTITY 2 208 1")
        objects32[93] = point402
        point403 = workPart.Points.FindObject("ENTITY 2 207 1")
        objects32[94] = point403
        point404 = workPart.Points.FindObject("ENTITY 2 206 1")
        objects32[95] = point404
        point405 = workPart.Points.FindObject("ENTITY 2 205 1")
        objects32[96] = point405
        point406 = workPart.Points.FindObject("ENTITY 2 204 1")
        objects32[97] = point406
        point407 = workPart.Points.FindObject("ENTITY 2 203 1")
        objects32[98] = point407
        point408 = workPart.Points.FindObject("ENTITY 2 202 1")
        objects32[99] = point408
        point409 = workPart.Points.FindObject("ENTITY 2 201 1")
        objects32[100] = point409
        point410 = workPart.Points.FindObject("ENTITY 2 200 1")
        objects32[101] = point410
        point411 = workPart.Points.FindObject("ENTITY 2 199 1")
        objects32[102] = point411
        point412 = workPart.Points.FindObject("ENTITY 2 198 1")
        objects32[103] = point412
        point413 = workPart.Points.FindObject("ENTITY 2 197 1")
        objects32[104] = point413
        point414 = workPart.Points.FindObject("ENTITY 2 196 1")
        objects32[105] = point414
        point415 = workPart.Points.FindObject("ENTITY 2 195 1")
        objects32[106] = point415
        point416 = workPart.Points.FindObject("ENTITY 2 194 1")
        objects32[107] = point416
        point417 = workPart.Points.FindObject("ENTITY 2 193 1")
        objects32[108] = point417
        point418 = workPart.Points.FindObject("ENTITY 2 192 1")
        objects32[109] = point418
        point419 = workPart.Points.FindObject("ENTITY 2 191 1")
        objects32[110] = point419
        point420 = workPart.Points.FindObject("ENTITY 2 190 1")
        objects32[111] = point420
        point421 = workPart.Points.FindObject("ENTITY 2 189 1")
        objects32[112] = point421
        point422 = workPart.Points.FindObject("ENTITY 2 188 1")
        objects32[113] = point422
        point423 = workPart.Points.FindObject("ENTITY 2 187 1")
        objects32[114] = point423
        point424 = workPart.Points.FindObject("ENTITY 2 186 1")
        objects32[115] = point424
        point425 = workPart.Points.FindObject("ENTITY 2 185 1")
        objects32[116] = point425
        point426 = workPart.Points.FindObject("ENTITY 2 184 1")
        objects32[117] = point426
        point427 = workPart.Points.FindObject("ENTITY 2 183 1")
        objects32[118] = point427
        point428 = workPart.Points.FindObject("ENTITY 2 182 1")
        objects32[119] = point428
        point429 = workPart.Points.FindObject("ENTITY 2 181 1")
        objects32[120] = point429
        point430 = workPart.Points.FindObject("ENTITY 2 180 1")
        objects32[121] = point430
        point431 = workPart.Points.FindObject("ENTITY 2 179 1")
        objects32[122] = point431
        point432 = workPart.Points.FindObject("ENTITY 2 178 1")
        objects32[123] = point432
        point433 = workPart.Points.FindObject("ENTITY 2 177 1")
        objects32[124] = point433
        point434 = workPart.Points.FindObject("ENTITY 2 176 1")
        objects32[125] = point434
        point435 = workPart.Points.FindObject("ENTITY 2 175 1")
        objects32[126] = point435
        point436 = workPart.Points.FindObject("ENTITY 2 174 1")
        objects32[127] = point436
        point437 = workPart.Points.FindObject("ENTITY 2 173 1")
        objects32[128] = point437
        point438 = workPart.Points.FindObject("ENTITY 2 172 1")
        objects32[129] = point438
        point439 = workPart.Points.FindObject("ENTITY 2 171 1")
        objects32[130] = point439
        point440 = workPart.Points.FindObject("ENTITY 2 170 1")
        objects32[131] = point440
        point441 = workPart.Points.FindObject("ENTITY 2 169 1")
        objects32[132] = point441
        point442 = workPart.Points.FindObject("ENTITY 2 168 1")
        objects32[133] = point442
        point443 = workPart.Points.FindObject("ENTITY 2 167 1")
        objects32[134] = point443
        point444 = workPart.Points.FindObject("ENTITY 2 166 1")
        objects32[135] = point444
        point445 = workPart.Points.FindObject("ENTITY 2 165 1")
        objects32[136] = point445
        point446 = workPart.Points.FindObject("ENTITY 2 164 1")
        objects32[137] = point446
        point447 = workPart.Points.FindObject("ENTITY 2 163 1")
        objects32[138] = point447
        point448 = workPart.Points.FindObject("ENTITY 2 162 1")
        objects32[139] = point448
        point449 = workPart.Points.FindObject("ENTITY 2 161 1")
        objects32[140] = point449
        point450 = workPart.Points.FindObject("ENTITY 2 160 1")
        objects32[141] = point450
        point451 = workPart.Points.FindObject("ENTITY 2 159 1")
        objects32[142] = point451
        point452 = workPart.Points.FindObject("ENTITY 2 158 1")
        objects32[143] = point452
        point453 = workPart.Points.FindObject("ENTITY 2 157 1")
        objects32[144] = point453
        point454 = workPart.Points.FindObject("ENTITY 2 156 1")
        objects32[145] = point454
        point455 = workPart.Points.FindObject("ENTITY 2 155 1")
        objects32[146] = point455
        point456 = workPart.Points.FindObject("ENTITY 2 154 1")
        objects32[147] = point456
        point457 = workPart.Points.FindObject("ENTITY 2 153 1")
        objects32[148] = point457
        point458 = workPart.Points.FindObject("ENTITY 2 152 1")
        objects32[149] = point458
        point459 = workPart.Points.FindObject("ENTITY 2 151 1")
        objects32[150] = point459
        point460 = workPart.Points.FindObject("ENTITY 2 150 1")
        objects32[151] = point460
        point461 = workPart.Points.FindObject("ENTITY 2 149 1")
        objects32[152] = point461
        point462 = workPart.Points.FindObject("ENTITY 2 148 1")
        objects32[153] = point462
        point463 = workPart.Points.FindObject("ENTITY 2 147 1")
        objects32[154] = point463
        point464 = workPart.Points.FindObject("ENTITY 2 146 1")
        objects32[155] = point464
        point465 = workPart.Points.FindObject("ENTITY 2 145 1")
        objects32[156] = point465
        point466 = workPart.Points.FindObject("ENTITY 2 144 1")
        objects32[157] = point466
        point467 = workPart.Points.FindObject("ENTITY 2 143 1")
        objects32[158] = point467
        point468 = workPart.Points.FindObject("ENTITY 2 142 1")
        objects32[159] = point468
        point469 = workPart.Points.FindObject("ENTITY 2 141 1")
        objects32[160] = point469
        point470 = workPart.Points.FindObject("ENTITY 2 140 1")
        objects32[161] = point470
        point471 = workPart.Points.FindObject("ENTITY 2 139 1")
        objects32[162] = point471
        point472 = workPart.Points.FindObject("ENTITY 2 138 1")
        objects32[163] = point472
        point473 = workPart.Points.FindObject("ENTITY 2 137 1")
        objects32[164] = point473
        point474 = workPart.Points.FindObject("ENTITY 2 136 1")
        objects32[165] = point474
        point475 = workPart.Points.FindObject("ENTITY 2 135 1")
        objects32[166] = point475
        point476 = workPart.Points.FindObject("ENTITY 2 134 1")
        objects32[167] = point476
        point477 = workPart.Points.FindObject("ENTITY 2 133 1")
        objects32[168] = point477
        point478 = workPart.Points.FindObject("ENTITY 2 132 1")
        objects32[169] = point478
        point479 = workPart.Points.FindObject("ENTITY 2 131 1")
        objects32[170] = point479
        point480 = workPart.Points.FindObject("ENTITY 2 130 1")
        objects32[171] = point480
        point481 = workPart.Points.FindObject("ENTITY 2 129 1")
        objects32[172] = point481
        point482 = workPart.Points.FindObject("ENTITY 2 128 1")
        objects32[173] = point482
        point483 = workPart.Points.FindObject("ENTITY 2 127 1")
        objects32[174] = point483
        point484 = workPart.Points.FindObject("ENTITY 2 126 1")
        objects32[175] = point484
        point485 = workPart.Points.FindObject("ENTITY 2 125 1")
        objects32[176] = point485
        point486 = workPart.Points.FindObject("ENTITY 2 124 1")
        objects32[177] = point486
        point487 = workPart.Points.FindObject("ENTITY 2 123 1")
        objects32[178] = point487
        point488 = workPart.Points.FindObject("ENTITY 2 122 1")
        objects32[179] = point488
        point489 = workPart.Points.FindObject("ENTITY 2 121 1")
        objects32[180] = point489
        point490 = workPart.Points.FindObject("ENTITY 2 120 1")
        objects32[181] = point490
        point491 = workPart.Points.FindObject("ENTITY 2 119 1")
        objects32[182] = point491
        point492 = workPart.Points.FindObject("ENTITY 2 118 1")
        objects32[183] = point492
        point493 = workPart.Points.FindObject("ENTITY 2 117 1")
        objects32[184] = point493
        point494 = workPart.Points.FindObject("ENTITY 2 116 1")
        objects32[185] = point494
        point495 = workPart.Points.FindObject("ENTITY 2 115 1")
        objects32[186] = point495
        point496 = workPart.Points.FindObject("ENTITY 2 114 1")
        objects32[187] = point496
        point497 = workPart.Points.FindObject("ENTITY 2 113 1")
        objects32[188] = point497
        point498 = workPart.Points.FindObject("ENTITY 2 112 1")
        objects32[189] = point498
        point499 = workPart.Points.FindObject("ENTITY 2 111 1")
        objects32[190] = point499
        point500 = workPart.Points.FindObject("ENTITY 2 110 1")
        objects32[191] = point500
        point501 = workPart.Points.FindObject("ENTITY 2 109 1")
        objects32[192] = point501
        point502 = workPart.Points.FindObject("ENTITY 2 108 1")
        objects32[193] = point502
        point503 = workPart.Points.FindObject("ENTITY 2 107 1")
        objects32[194] = point503
        point504 = workPart.Points.FindObject("ENTITY 2 106 1")
        objects32[195] = point504
        point505 = workPart.Points.FindObject("ENTITY 2 105 1")
        objects32[196] = point505
        point506 = workPart.Points.FindObject("ENTITY 2 104 1")
        objects32[197] = point506
        point507 = workPart.Points.FindObject("ENTITY 2 103 1")
        objects32[198] = point507
        point508 = workPart.Points.FindObject("ENTITY 2 102 1")
        objects32[199] = point508
        point509 = workPart.Points.FindObject("ENTITY 2 101 1")
        objects32[200] = point509
        point510 = workPart.Points.FindObject("ENTITY 2 100 1")
        objects32[201] = point510
        point511 = workPart.Points.FindObject("ENTITY 2 99 1")
        objects32[202] = point511
        point512 = workPart.Points.FindObject("ENTITY 2 98 1")
        objects32[203] = point512
        point513 = workPart.Points.FindObject("ENTITY 2 97 1")
        objects32[204] = point513
        point514 = workPart.Points.FindObject("ENTITY 2 96 1")
        objects32[205] = point514
        point515 = workPart.Points.FindObject("ENTITY 2 95 1")
        objects32[206] = point515
        point516 = workPart.Points.FindObject("ENTITY 2 94 1")
        objects32[207] = point516
        point517 = workPart.Points.FindObject("ENTITY 2 93 1")
        objects32[208] = point517
        point518 = workPart.Points.FindObject("ENTITY 2 92 1")
        objects32[209] = point518
        point519 = workPart.Points.FindObject("ENTITY 2 91 1")
        objects32[210] = point519
        point520 = workPart.Points.FindObject("ENTITY 2 90 1")
        objects32[211] = point520
        point521 = workPart.Points.FindObject("ENTITY 2 89 1")
        objects32[212] = point521
        point522 = workPart.Points.FindObject("ENTITY 2 88 1")
        objects32[213] = point522
        point523 = workPart.Points.FindObject("ENTITY 2 87 1")
        objects32[214] = point523
        point524 = workPart.Points.FindObject("ENTITY 2 86 1")
        objects32[215] = point524
        point525 = workPart.Points.FindObject("ENTITY 2 85 1")
        objects32[216] = point525
        point526 = workPart.Points.FindObject("ENTITY 2 84 1")
        objects32[217] = point526
        point527 = workPart.Points.FindObject("ENTITY 2 83 1")
        objects32[218] = point527
        point528 = workPart.Points.FindObject("ENTITY 2 82 1")
        objects32[219] = point528
        point529 = workPart.Points.FindObject("ENTITY 2 81 1")
        objects32[220] = point529
        point530 = workPart.Points.FindObject("ENTITY 2 80 1")
        objects32[221] = point530
        point531 = workPart.Points.FindObject("ENTITY 2 79 1")
        objects32[222] = point531
        point532 = workPart.Points.FindObject("ENTITY 2 78 1")
        objects32[223] = point532
        point533 = workPart.Points.FindObject("ENTITY 2 77 1")
        objects32[224] = point533
        point534 = workPart.Points.FindObject("ENTITY 2 76 1")
        objects32[225] = point534
        point535 = workPart.Points.FindObject("ENTITY 2 75 1")
        objects32[226] = point535
        point536 = workPart.Points.FindObject("ENTITY 2 74 1")
        objects32[227] = point536
        point537 = workPart.Points.FindObject("ENTITY 2 73 1")
        objects32[228] = point537
        point538 = workPart.Points.FindObject("ENTITY 2 72 1")
        objects32[229] = point538
        point539 = workPart.Points.FindObject("ENTITY 2 71 1")
        objects32[230] = point539
        point540 = workPart.Points.FindObject("ENTITY 2 70 1")
        objects32[231] = point540
        point541 = workPart.Points.FindObject("ENTITY 2 69 1")
        objects32[232] = point541
        point542 = workPart.Points.FindObject("ENTITY 2 68 1")
        objects32[233] = point542
        point543 = workPart.Points.FindObject("ENTITY 2 67 1")
        objects32[234] = point543
        point544 = workPart.Points.FindObject("ENTITY 2 66 1")
        objects32[235] = point544
        point545 = workPart.Points.FindObject("ENTITY 2 65 1")
        objects32[236] = point545
        point546 = workPart.Points.FindObject("ENTITY 2 64 1")
        objects32[237] = point546
        point547 = workPart.Points.FindObject("ENTITY 2 63 1")
        objects32[238] = point547
        point548 = workPart.Points.FindObject("ENTITY 2 62 1")
        objects32[239] = point548
        point549 = workPart.Points.FindObject("ENTITY 2 61 1")
        objects32[240] = point549
        point550 = workPart.Points.FindObject("ENTITY 2 60 1")
        objects32[241] = point550
        point551 = workPart.Points.FindObject("ENTITY 2 59 1")
        objects32[242] = point551
        point552 = workPart.Points.FindObject("ENTITY 2 58 1")
        objects32[243] = point552
        point553 = workPart.Points.FindObject("ENTITY 2 57 1")
        objects32[244] = point553
        point554 = workPart.Points.FindObject("ENTITY 2 56 1")
        objects32[245] = point554
        point555 = workPart.Points.FindObject("ENTITY 2 55 1")
        objects32[246] = point555
        point556 = workPart.Points.FindObject("ENTITY 2 54 1")
        objects32[247] = point556
        point557 = workPart.Points.FindObject("ENTITY 2 53 1")
        objects32[248] = point557
        point558 = workPart.Points.FindObject("ENTITY 2 52 1")
        objects32[249] = point558
        point559 = workPart.Points.FindObject("ENTITY 2 51 1")
        objects32[250] = point559
        point560 = workPart.Points.FindObject("ENTITY 2 50 1")
        objects32[251] = point560
        point561 = workPart.Points.FindObject("ENTITY 2 49 1")
        objects32[252] = point561
        point562 = workPart.Points.FindObject("ENTITY 2 48 1")
        objects32[253] = point562
        point563 = workPart.Points.FindObject("ENTITY 2 47 1")
        objects32[254] = point563
        point564 = workPart.Points.FindObject("ENTITY 2 46 1")
        objects32[255] = point564
        point565 = workPart.Points.FindObject("ENTITY 2 45 1")
        objects32[256] = point565
        point566 = workPart.Points.FindObject("ENTITY 2 44 1")
        objects32[257] = point566
        point567 = workPart.Points.FindObject("ENTITY 2 43 1")
        objects32[258] = point567
        point568 = workPart.Points.FindObject("ENTITY 2 42 1")
        objects32[259] = point568
        point569 = workPart.Points.FindObject("ENTITY 2 41 1")
        objects32[260] = point569
        point570 = workPart.Points.FindObject("ENTITY 2 40 1")
        objects32[261] = point570
        point571 = workPart.Points.FindObject("ENTITY 2 39 1")
        objects32[262] = point571
        point572 = workPart.Points.FindObject("ENTITY 2 38 1")
        objects32[263] = point572
        point573 = workPart.Points.FindObject("ENTITY 2 37 1")
        objects32[264] = point573
        point574 = workPart.Points.FindObject("ENTITY 2 36 1")
        objects32[265] = point574
        point575 = workPart.Points.FindObject("ENTITY 2 35 1")
        objects32[266] = point575
        point576 = workPart.Points.FindObject("ENTITY 2 34 1")
        objects32[267] = point576
        point577 = workPart.Points.FindObject("ENTITY 2 33 1")
        objects32[268] = point577
        point578 = workPart.Points.FindObject("ENTITY 2 32 1")
        objects32[269] = point578
        point579 = workPart.Points.FindObject("ENTITY 2 31 1")
        objects32[270] = point579
        point580 = workPart.Points.FindObject("ENTITY 2 30 1")
        objects32[271] = point580
        point581 = workPart.Points.FindObject("ENTITY 2 29 1")
        objects32[272] = point581
        point582 = workPart.Points.FindObject("ENTITY 2 28 1")
        objects32[273] = point582
        point583 = workPart.Points.FindObject("ENTITY 2 27 1")
        objects32[274] = point583
        point584 = workPart.Points.FindObject("ENTITY 2 26 1")
        objects32[275] = point584
        point585 = workPart.Points.FindObject("ENTITY 2 25 1")
        objects32[276] = point585
        point586 = workPart.Points.FindObject("ENTITY 2 24 1")
        objects32[277] = point586
        point587 = workPart.Points.FindObject("ENTITY 2 23 1")
        objects32[278] = point587
        point588 = workPart.Points.FindObject("ENTITY 2 22 1")
        objects32[279] = point588
        point589 = workPart.Points.FindObject("ENTITY 2 21 1")
        objects32[280] = point589
        point590 = workPart.Points.FindObject("ENTITY 2 20 1")
        objects32[281] = point590
        point591 = workPart.Points.FindObject("ENTITY 2 19 1")
        objects32[282] = point591
        point592 = workPart.Points.FindObject("ENTITY 2 18 1")
        objects32[283] = point592
        point593 = workPart.Points.FindObject("ENTITY 2 17 1")
        objects32[284] = point593
        point594 = workPart.Points.FindObject("ENTITY 2 16 1")
        objects32[285] = point594
        point595 = workPart.Points.FindObject("ENTITY 2 15 1")
        objects32[286] = point595
        point596 = workPart.Points.FindObject("ENTITY 2 14 1")
        objects32[287] = point596
        point597 = workPart.Points.FindObject("ENTITY 2 13 1")
        objects32[288] = point597
        point598 = workPart.Points.FindObject("ENTITY 2 12 1")
        objects32[289] = point598
        point599 = workPart.Points.FindObject("ENTITY 2 11 1")
        objects32[290] = point599
        point600 = workPart.Points.FindObject("ENTITY 2 10 1")
        objects32[291] = point600
        point601 = workPart.Points.FindObject("ENTITY 2 9 1")
        objects32[292] = point601
        point602 = workPart.Points.FindObject("ENTITY 2 8 1")
        objects32[293] = point602
        point603 = workPart.Points.FindObject("ENTITY 2 7 1")
        objects32[294] = point603
        point604 = workPart.Points.FindObject("ENTITY 2 6 1")
        objects32[295] = point604
        point605 = workPart.Points.FindObject("ENTITY 2 5 1")
        objects32[296] = point605
        point606 = workPart.Points.FindObject("ENTITY 2 4 1")
        objects32[297] = point606
        point607 = workPart.Points.FindObject("ENTITY 2 3 1")
        objects32[298] = point607
        point608 = workPart.Points.FindObject("ENTITY 2 2 1")
        objects32[299] = point608
        point609 = workPart.Points.FindObject("ENTITY 2 1 1")
        objects32[300] = point609
        group2 = nXObject9
        objects32[301] = group2
        added3 = fitCurveBuilder2.Target.Add(objects32)
        
        geometricConstraintData3 = fitCurveBuilder2.ConstraintManager.FindItem(0)
        
        point610 = geometricConstraintData3.Point
        
        geometricConstraintData4 = fitCurveBuilder2.ConstraintManager.FindItem(1)
        
        point611 = geometricConstraintData4.Point
        
        theSession.SetUndoMarkName(markId57, "拟合曲线 - 选择")
        
        theSession.SetUndoMarkVisibility(markId57, None, NXOpen.Session.MarkVisibility.Visible)
        
        theSession.SetUndoMarkVisibility(markId55, None, NXOpen.Session.MarkVisibility.Invisible)
        
        markId58 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        theSession.DeleteUndoMark(markId58, None)
        
        markId59 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        fitCurveBuilder2.HasReversedDirection = True
        
        theSession.SetUndoMarkName(markId59, "拟合曲线 - 反向")
        
        theSession.SetUndoMarkVisibility(markId59, None, NXOpen.Session.MarkVisibility.Visible)
        
        theSession.SetUndoMarkVisibility(markId55, None, NXOpen.Session.MarkVisibility.Invisible)
        
        markId60 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        markId61 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        theSession.DeleteUndoMark(markId61, None)
        
        markId62 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "拟合曲线")
        
        nXObject10 = fitCurveBuilder2.Commit()
        
        theSession.DeleteUndoMark(markId62, None)
        
        theSession.SetUndoMarkName(markId55, "拟合曲线")
        
        fitCurveBuilder2.Destroy()
        
        theSession.SetUndoMarkVisibility(markId55, None, NXOpen.Session.MarkVisibility.Visible)
        
        theSession.DeleteUndoMark(markId59, None)
        
        theSession.DeleteUndoMark(markId57, None)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->5 抽取几何特征
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->关联复制(A)->抽取几何特征(E)...
        # ----------------------------------------------
        markId63 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        wavePointBuilder3 = workPart.Features.CreateWavePointBuilder(NXOpen.Features.Feature.Null)
        
        waveDatumBuilder3 = workPart.Features.CreateWaveDatumBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder3 = workPart.Features.CreateCompositeCurveBuilder(NXOpen.Features.Feature.Null)
        
        extractFaceBuilder3 = workPart.Features.CreateExtractFaceBuilder(NXOpen.Features.Feature.Null)
        
        mirrorBodyBuilder3 = workPart.Features.CreateMirrorBodyBuilder(NXOpen.Features.Feature.Null)
        
        waveSketchBuilder3 = workPart.Features.CreateWaveSketchBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder3.CurveFitData.Tolerance = 0.001
        
        compositeCurveBuilder3.CurveFitData.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder3.Section.SetAllowRefCrvs(False)
        
        extractFaceBuilder3.FaceOption = NXOpen.Features.ExtractFaceBuilder.FaceOptionType.AdjacentFaces
        
        compositeCurveBuilder3.Associative = False
        
        waveDatumBuilder3.ParentPart = NXOpen.Features.WaveDatumBuilder.ParentPartType.WorkPart
        
        wavePointBuilder3.ParentPart = NXOpen.Features.WavePointBuilder.ParentPartType.WorkPart
        
        extractFaceBuilder3.ParentPart = NXOpen.Features.ExtractFaceBuilder.ParentPartType.WorkPart
        
        mirrorBodyBuilder3.ParentPartType = NXOpen.Features.MirrorBodyBuilder.ParentPart.WorkPart
        
        compositeCurveBuilder3.ParentPart = NXOpen.Features.CompositeCurveBuilder.PartType.WorkPart
        
        waveSketchBuilder3.ParentPart = NXOpen.Features.WaveSketchBuilder.ParentPartType.WorkPart
        
        compositeCurveBuilder3.Associative = False
        
        theSession.SetUndoMarkName(markId63, "抽取几何特征 对话框")
        
        compositeCurveBuilder3.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder3.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder3.Section.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder3.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder3.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder3.Associative = False
        
        compositeCurveBuilder3.FixAtCurrentTimestamp = False
        
        compositeCurveBuilder3.HideOriginal = False
        
        compositeCurveBuilder3.InheritDisplayProperties = False
        
        compositeCurveBuilder3.Section.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.CurvesAndPoints)
        
        waveDatumBuilder3.Associative = False
        
        waveDatumBuilder3.HideOriginal = False
        
        waveDatumBuilder3.InheritDisplayProperties = False
        
        compositeCurveBuilder3.Associative = False
        
        compositeCurveBuilder3.FixAtCurrentTimestamp = False
        
        compositeCurveBuilder3.HideOriginal = False
        
        compositeCurveBuilder3.InheritDisplayProperties = False
        
        markId64 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId65 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions5 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions5.SetSelectedFromInactive(False)
        
        points1 = [NXOpen.Point.Null] * 1 
        points1[0] = point334
        curveDumbRule3 = workPart.ScRuleFactory.CreateRuleCurveDumbFromPoints(points1, selectionIntentRuleOptions5)
        
        selectionIntentRuleOptions5.Dispose()
        compositeCurveBuilder3.Section.AllowSelfIntersection(False)
        
        compositeCurveBuilder3.Section.AllowDegenerateCurves(False)
        
        rules5 = [None] * 1 
        rules5[0] = curveDumbRule3
        helpPoint5 = NXOpen.Point3d(42.70461094379425, 0.36206639208830893, -0.11423638463020325)
        compositeCurveBuilder3.Section.AddToSection(rules5, point334, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint5, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId65, None)
        
        expression32 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        theSession.DeleteUndoMark(markId64, None)
        
        markId66 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId67 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        compositeCurveBuilder3.Section.RemoveSingleSectionElement(point334, NXOpen.Section.Mode.Create)
        
        theSession.DeleteUndoMark(markId67, None)
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression32)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        theSession.DeleteUndoMark(markId66, None)
        
        markId68 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId69 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions6 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions6.SetSelectedFromInactive(False)
        
        curves3 = [NXOpen.IBaseCurve.Null] * 1 
        spline3 = workPart.Splines.FindObject("ENTITY 9 1 1")
        curves3[0] = spline3
        curveDumbRule4 = workPart.ScRuleFactory.CreateRuleBaseCurveDumb(curves3, selectionIntentRuleOptions6)
        
        selectionIntentRuleOptions6.Dispose()
        compositeCurveBuilder3.Section.AllowSelfIntersection(False)
        
        compositeCurveBuilder3.Section.AllowDegenerateCurves(False)
        
        rules6 = [None] * 1 
        rules6[0] = curveDumbRule4
        helpPoint6 = NXOpen.Point3d(39.979537845472741, 0.33899414048196741, -0.10769197257069446)
        compositeCurveBuilder3.Section.AddToSection(rules6, spline3, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint6, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId69, None)
        
        theSession.DeleteUndoMark(markId68, None)
        
        markId70 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        theSession.DeleteUndoMark(markId70, None)
        
        markId71 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        nXObject11 = compositeCurveBuilder3.Commit()
        
        theSession.DeleteUndoMark(markId71, None)
        
        theSession.SetUndoMarkName(markId63, "抽取几何特征")
        
        compositeCurveBuilder3.Destroy()
        
        waveDatumBuilder3.Destroy()
        
        wavePointBuilder3.Destroy()
        
        extractFaceBuilder3.Destroy()
        
        mirrorBodyBuilder3.Destroy()
        
        waveSketchBuilder3.Destroy()
        
        markId72 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects33 = [NXOpen.DisplayableObject.Null] * 1 
        compositeCurve3 = nXObject11
        spline4 = compositeCurve3.FindObject("CURVE 1")
        objects33[0] = spline4
        theSession.DisplayManager.BlankObjects(objects33)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：编辑(E)->选择(L)->全选(S)
        # ----------------------------------------------
        # Refer to the sample NXOpen application, Selection for "Select All" alternatives.
        # ----------------------------------------------
        #   菜单：编辑(E)->删除(D)...
        # ----------------------------------------------
        markId73 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Delete")
        
        theSession.UpdateManager.ClearErrorList()
        
        markId74 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Delete")
        
        objects34 = [NXOpen.TaggedObject.Null] * 303 
        objects34[0] = point537
        objects34[1] = point538
        objects34[2] = point539
        objects34[3] = point540
        objects34[4] = point541
        objects34[5] = point542
        objects34[6] = point543
        objects34[7] = point544
        objects34[8] = point545
        objects34[9] = point546
        objects34[10] = point547
        objects34[11] = point548
        objects34[12] = point549
        objects34[13] = point550
        objects34[14] = point551
        objects34[15] = point552
        objects34[16] = point553
        objects34[17] = point554
        objects34[18] = point555
        objects34[19] = point556
        objects34[20] = point557
        objects34[21] = point558
        objects34[22] = point559
        objects34[23] = point560
        objects34[24] = point561
        objects34[25] = point562
        objects34[26] = point563
        objects34[27] = point564
        objects34[28] = point565
        objects34[29] = point566
        objects34[30] = point567
        objects34[31] = point568
        objects34[32] = point569
        objects34[33] = point570
        objects34[34] = point571
        objects34[35] = point572
        objects34[36] = point573
        objects34[37] = point457
        objects34[38] = point458
        objects34[39] = point459
        objects34[40] = point460
        objects34[41] = point461
        objects34[42] = point462
        objects34[43] = point463
        objects34[44] = point464
        objects34[45] = point465
        objects34[46] = point466
        objects34[47] = point467
        objects34[48] = point468
        objects34[49] = point469
        objects34[50] = point470
        objects34[51] = point471
        objects34[52] = point472
        objects34[53] = point473
        objects34[54] = point474
        objects34[55] = point475
        objects34[56] = point476
        objects34[57] = point477
        objects34[58] = point478
        objects34[59] = point479
        objects34[60] = point480
        objects34[61] = point481
        objects34[62] = point482
        objects34[63] = point483
        objects34[64] = point484
        objects34[65] = point485
        objects34[66] = point486
        objects34[67] = point487
        objects34[68] = point488
        objects34[69] = point489
        objects34[70] = point490
        objects34[71] = point491
        objects34[72] = point492
        objects34[73] = point493
        objects34[74] = point494
        objects34[75] = point495
        objects34[76] = point496
        objects34[77] = point497
        objects34[78] = point498
        objects34[79] = point499
        objects34[80] = point500
        objects34[81] = point501
        objects34[82] = point502
        objects34[83] = point368
        objects34[84] = point369
        objects34[85] = point370
        objects34[86] = point371
        objects34[87] = point372
        objects34[88] = point373
        objects34[89] = point374
        objects34[90] = point375
        objects34[91] = point376
        objects34[92] = point377
        objects34[93] = point378
        objects34[94] = point379
        objects34[95] = point380
        objects34[96] = point381
        objects34[97] = point382
        objects34[98] = point383
        objects34[99] = point384
        objects34[100] = point385
        objects34[101] = point386
        objects34[102] = point387
        objects34[103] = point388
        objects34[104] = point389
        objects34[105] = point390
        objects34[106] = point391
        objects34[107] = point392
        objects34[108] = point393
        objects34[109] = point394
        objects34[110] = point395
        objects34[111] = point396
        objects34[112] = point397
        objects34[113] = point398
        objects34[114] = point399
        objects34[115] = point400
        objects34[116] = point401
        objects34[117] = point402
        objects34[118] = point403
        objects34[119] = point404
        objects34[120] = point405
        objects34[121] = point406
        objects34[122] = point407
        objects34[123] = point408
        objects34[124] = point409
        objects34[125] = point410
        objects34[126] = point411
        objects34[127] = point412
        objects34[128] = point413
        objects34[129] = point414
        objects34[130] = point415
        objects34[131] = point416
        objects34[132] = point417
        objects34[133] = point418
        objects34[134] = point419
        objects34[135] = point420
        objects34[136] = point341
        objects34[137] = point342
        objects34[138] = point343
        objects34[139] = point344
        objects34[140] = point345
        objects34[141] = point346
        objects34[142] = point347
        objects34[143] = point348
        objects34[144] = point349
        objects34[145] = point350
        objects34[146] = point309
        objects34[147] = point310
        objects34[148] = point311
        objects34[149] = point312
        objects34[150] = point313
        objects34[151] = point314
        objects34[152] = point315
        objects34[153] = point316
        objects34[154] = point317
        objects34[155] = point318
        objects34[156] = point319
        objects34[157] = point320
        objects34[158] = point321
        objects34[159] = point322
        objects34[160] = point323
        objects34[161] = point324
        objects34[162] = point325
        objects34[163] = point326
        objects34[164] = point327
        objects34[165] = point328
        objects34[166] = point329
        objects34[167] = point330
        objects34[168] = point331
        objects34[169] = point332
        objects34[170] = point333
        objects34[171] = point334
        objects34[172] = point335
        objects34[173] = point336
        objects34[174] = point337
        objects34[175] = point338
        objects34[176] = point339
        objects34[177] = point340
        objects34[178] = spline3
        objects34[179] = point574
        objects34[180] = point575
        objects34[181] = point576
        objects34[182] = point577
        objects34[183] = point578
        objects34[184] = point579
        objects34[185] = point580
        objects34[186] = point581
        objects34[187] = point582
        objects34[188] = point583
        objects34[189] = point584
        objects34[190] = point585
        objects34[191] = point586
        objects34[192] = point587
        objects34[193] = point588
        objects34[194] = point589
        objects34[195] = point590
        objects34[196] = point591
        objects34[197] = point592
        objects34[198] = point593
        objects34[199] = point594
        objects34[200] = point595
        objects34[201] = point596
        objects34[202] = point597
        objects34[203] = point598
        objects34[204] = point599
        objects34[205] = point600
        objects34[206] = point601
        objects34[207] = point602
        objects34[208] = point603
        objects34[209] = point604
        objects34[210] = point605
        objects34[211] = point606
        objects34[212] = point607
        objects34[213] = point608
        objects34[214] = point609
        objects34[215] = group2
        objects34[216] = point503
        objects34[217] = point504
        objects34[218] = point505
        objects34[219] = point506
        objects34[220] = point507
        objects34[221] = point508
        objects34[222] = point509
        objects34[223] = point510
        objects34[224] = point511
        objects34[225] = point512
        objects34[226] = point513
        objects34[227] = point514
        objects34[228] = point515
        objects34[229] = point516
        objects34[230] = point517
        objects34[231] = point518
        objects34[232] = point519
        objects34[233] = point520
        objects34[234] = point521
        objects34[235] = point522
        objects34[236] = point523
        objects34[237] = point524
        objects34[238] = point525
        objects34[239] = point526
        objects34[240] = point527
        objects34[241] = point528
        objects34[242] = point529
        objects34[243] = point530
        objects34[244] = point531
        objects34[245] = point532
        objects34[246] = point533
        objects34[247] = point534
        objects34[248] = point535
        objects34[249] = point536
        objects34[250] = point421
        objects34[251] = point422
        objects34[252] = point423
        objects34[253] = point424
        objects34[254] = point425
        objects34[255] = point426
        objects34[256] = point427
        objects34[257] = point428
        objects34[258] = point429
        objects34[259] = point430
        objects34[260] = point431
        objects34[261] = point432
        objects34[262] = point433
        objects34[263] = point434
        objects34[264] = point435
        objects34[265] = point436
        objects34[266] = point437
        objects34[267] = point438
        objects34[268] = point439
        objects34[269] = point440
        objects34[270] = point441
        objects34[271] = point442
        objects34[272] = point443
        objects34[273] = point444
        objects34[274] = point445
        objects34[275] = point446
        objects34[276] = point447
        objects34[277] = point448
        objects34[278] = point449
        objects34[279] = point450
        objects34[280] = point451
        objects34[281] = point452
        objects34[282] = point453
        objects34[283] = point454
        objects34[284] = point455
        objects34[285] = point456
        objects34[286] = point351
        objects34[287] = point352
        objects34[288] = point353
        objects34[289] = point354
        objects34[290] = point355
        objects34[291] = point356
        objects34[292] = point357
        objects34[293] = point358
        objects34[294] = point359
        objects34[295] = point360
        objects34[296] = point361
        objects34[297] = point362
        objects34[298] = point363
        objects34[299] = point364
        objects34[300] = point365
        objects34[301] = point366
        objects34[302] = point367
        nErrs8 = theSession.UpdateManager.AddObjectsToDeleteList(objects34)
        
        id3 = theSession.NewestVisibleUndoMark
        
        nErrs9 = theSession.UpdateManager.DoUpdate(id3)
        
        theSession.DeleteUndoMark(markId73, None)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->5 扫掠
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->扫掠(W)->扫掠(S)...
        # ----------------------------------------------
        markId75 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        sweptBuilder2 = workPart.Features.CreateSweptBuilder(NXOpen.Features.Swept.Null)
        
        expression33 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        expression34 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        expression35 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        sweptBuilder2.G0Tolerance = 0.001
        
        sweptBuilder2.G1Tolerance = 0.050000000000000003
        
        sweptBuilder2.OrientationMethod.AngularLaw.Value.SetFormula("0")
        
        sweptBuilder2.OrientationMethod.AngularLaw.StartValue.SetFormula("0")
        
        sweptBuilder2.OrientationMethod.AngularLaw.EndValue.SetFormula("0")
        
        sweptBuilder2.ScalingMethod.AreaLaw.Value.SetFormula("1")
        
        sweptBuilder2.ScalingMethod.AreaLaw.StartValue.SetFormula("1")
        
        sweptBuilder2.ScalingMethod.AreaLaw.EndValue.SetFormula("1")
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.Value.SetFormula("1")
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.StartValue.SetFormula("1")
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.EndValue.SetFormula("1")
        
        theSession.SetUndoMarkName(markId75, "扫掠 对话框")
        
        sweptBuilder2.Spine.DistanceTolerance = 0.001
        
        sweptBuilder2.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder2.AlignmentMethod.AlignCurve.DistanceTolerance = 0.001
        
        sweptBuilder2.AlignmentMethod.AlignCurve.ChainingTolerance = 0.00095
        
        sweptBuilder2.OrientationMethod.OrientationCurve.DistanceTolerance = 0.001
        
        sweptBuilder2.OrientationMethod.OrientationCurve.ChainingTolerance = 0.00095
        
        sweptBuilder2.OrientationMethod.AngularLaw.AlongSpineData.Spine.DistanceTolerance = 0.001
        
        sweptBuilder2.OrientationMethod.AngularLaw.AlongSpineData.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder2.OrientationMethod.AngularLaw.LawCurve.DistanceTolerance = 0.001
        
        sweptBuilder2.OrientationMethod.AngularLaw.LawCurve.ChainingTolerance = 0.00095
        
        sweptBuilder2.ScalingMethod.ScalingCurve.DistanceTolerance = 0.001
        
        sweptBuilder2.ScalingMethod.ScalingCurve.ChainingTolerance = 0.00095
        
        sweptBuilder2.ScalingMethod.AreaLaw.AlongSpineData.Spine.DistanceTolerance = 0.001
        
        sweptBuilder2.ScalingMethod.AreaLaw.AlongSpineData.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder2.ScalingMethod.AreaLaw.LawCurve.DistanceTolerance = 0.001
        
        sweptBuilder2.ScalingMethod.AreaLaw.LawCurve.ChainingTolerance = 0.00095
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.AlongSpineData.Spine.DistanceTolerance = 0.001
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.AlongSpineData.Spine.ChainingTolerance = 0.00095
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.LawCurve.DistanceTolerance = 0.001
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.LawCurve.ChainingTolerance = 0.00095
        
        sweptBuilder2.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.AlignmentMethod.AlignCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.OrientationMethod.OrientationCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.OrientationMethod.AngularLaw.AlongSpineData.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.OrientationMethod.AngularLaw.LawCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.ScalingMethod.ScalingCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.ScalingMethod.AreaLaw.AlongSpineData.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.ScalingMethod.AreaLaw.LawCurve.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.AlongSpineData.Spine.AngleTolerance = 0.050000000000000003
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.LawCurve.AngleTolerance = 0.050000000000000003
        
        section3 = workPart.Sections.CreateSection(0.00095, 0.001, 0.050000000000000003)
        
        sweptBuilder2.SectionList.Append(section3)
        
        section3.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.OnlyCurves)
        
        markId76 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId77 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions7 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions7.SetSelectedFromInactive(False)
        
        features3 = [NXOpen.Features.Feature.Null] * 1 
        features3[0] = compositeCurve1
        curveFeatureRule3 = workPart.ScRuleFactory.CreateRuleCurveFeature(features3, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions7)
        
        selectionIntentRuleOptions7.Dispose()
        section3.AllowSelfIntersection(False)
        
        section3.AllowDegenerateCurves(False)
        
        rules7 = [None] * 1 
        rules7[0] = curveFeatureRule3
        helpPoint7 = NXOpen.Point3d(0.0, 0.0, 0.0)
        section3.AddToSection(rules7, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint7, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId77, None)
        
        sections2 = [NXOpen.Section.Null] * 1 
        sections2[0] = section3
        sweptBuilder2.AlignmentMethod.SetSections(sections2)
        
        expression36 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        theSession.DeleteUndoMark(markId76, None)
        
        section3.ReverseDirectionOfLoop(0)
        
        try:
            sweptBuilder2.AlignmentMethod.UpdateSectionAtIndex(0)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(1)
            
        markId78 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId79 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions8 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions8.SetSelectedFromInactive(False)
        
        features4 = [NXOpen.Features.Feature.Null] * 1 
        features4[0] = compositeCurve3
        curveFeatureRule4 = workPart.ScRuleFactory.CreateRuleCurveFeature(features4, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions8)
        
        selectionIntentRuleOptions8.Dispose()
        section3.AllowSelfIntersection(False)
        
        section3.AllowDegenerateCurves(False)
        
        rules8 = [None] * 1 
        rules8[0] = curveFeatureRule4
        helpPoint8 = NXOpen.Point3d(0.0, 0.0, 0.0)
        section3.AddToSection(rules8, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint8, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId79, None)
        
        try:
            # 无法从有多个环的轮廓创建单条曲线。
            sweptBuilder2.AlignmentMethod.UpdateSectionAtIndex(0)
        except NXOpen.NXException as ex:
            ex.AssertErrorCode(672082)
            
        sweptBuilder2.AlignmentMethod.UnloadSections()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression36)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        expression37 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        theSession.DeleteUndoMark(markId78, None)
        
        markId80 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId81 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        section3.RemoveRules(spline4, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.Section.Mode.Create)
        
        theSession.DeleteUndoMark(markId81, None)
        
        sections3 = [NXOpen.Section.Null] * 1 
        sections3[0] = section3
        sweptBuilder2.AlignmentMethod.SetSections(sections3)
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression37)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        expression38 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        theSession.DeleteUndoMark(markId80, None)
        
        section4 = workPart.Sections.CreateSection(0.00095, 0.001, 0.050000000000000003)
        
        sweptBuilder2.GuideList.Append(section4)
        
        section4.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.OnlyCurves)
        
        markId82 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId83 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions9 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions9.SetSelectedFromInactive(False)
        
        features5 = [NXOpen.Features.Feature.Null] * 1 
        features5[0] = compositeCurve3
        curveFeatureRule5 = workPart.ScRuleFactory.CreateRuleCurveFeature(features5, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions9)
        
        selectionIntentRuleOptions9.Dispose()
        section4.AllowSelfIntersection(False)
        
        section4.AllowDegenerateCurves(False)
        
        rules9 = [None] * 1 
        rules9[0] = curveFeatureRule5
        helpPoint9 = NXOpen.Point3d(0.0, 0.0, 0.0)
        section4.AddToSection(rules9, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint9, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId83, None)
        
        sweptBuilder2.ScalingMethod.AreaLaw.AlongSpineData.SetFeatureSpine(section4)
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.AlongSpineData.SetFeatureSpine(section4)
        
        markId84 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId84, "Update Law Data", False)
        
        markId85 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId85, "Update Law Data", False)
        
        sweptBuilder2.OrientationMethod.AngularLaw.AlongSpineData.SetFeatureSpine(section4)
        
        markId86 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId86, "Update Law Data", False)
        
        theSession.DeleteUndoMark(markId82, None)
        
        markId87 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "扫掠")
        
        theSession.DeleteUndoMark(markId87, None)
        
        markId88 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "扫掠")
        
        nXObject12 = sweptBuilder2.Commit()
        
        displayModification2 = theSession.DisplayManager.NewDisplayModification()
        
        displayModification2.ApplyToAllFaces = False
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects35 = [NXOpen.DisplayableObject.Null] * 1 
        swept2 = nXObject12
        face23 = swept2.FindObject("FACE 10001 {(256.5165364186971,4.5766680150801,1.0957800391369) SWEPT(5)}")
        objects35[0] = face23
        displayModification2.Apply(objects35)
        
        face23.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects36 = [NXOpen.DisplayableObject.Null] * 1 
        face24 = swept2.FindObject("FACE 1 {(0,0.9062566580345,3.6078924041855) SWEPT(5)}")
        objects36[0] = face24
        displayModification2.Apply(objects36)
        
        face24.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects37 = [NXOpen.DisplayableObject.Null] * 1 
        face25 = swept2.FindObject("FACE 10002 {(256.225983888424,7.8931722842321,1.9475900086277) SWEPT(5)}")
        objects37[0] = face25
        displayModification2.Apply(objects37)
        
        face25.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects38 = [NXOpen.DisplayableObject.Null] * 1 
        face26 = swept2.FindObject("FACE 2 {(508.1824338932621,52.9282991569974,-11.1457375853668) SWEPT(5)}")
        objects38[0] = face26
        displayModification2.Apply(objects38)
        
        face26.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects39 = [NXOpen.DisplayableObject.Null] * 1 
        face27 = swept2.FindObject("FACE 10020 {(256.6287001079312,3.4427827508442,1.2741847492591) SWEPT(5)}")
        objects39[0] = face27
        displayModification2.Apply(objects39)
        
        face27.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects40 = [NXOpen.DisplayableObject.Null] * 1 
        face28 = swept2.FindObject("FACE 10003 {(255.8940590373642,11.4333390462786,2.0594548273365) SWEPT(5)}")
        objects40[0] = face28
        displayModification2.Apply(objects40)
        
        face28.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects41 = [NXOpen.DisplayableObject.Null] * 1 
        face29 = swept2.FindObject("FACE 10004 {(255.80225480623,11.7098965270131,-0.3437958061709) SWEPT(5)}")
        objects41[0] = face29
        displayModification2.Apply(objects41)
        
        face29.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects42 = [NXOpen.DisplayableObject.Null] * 1 
        face30 = swept2.FindObject("FACE 10005 {(255.6416502791565,12.5407396148561,-3.3457869993159) SWEPT(5)}")
        objects42[0] = face30
        displayModification2.Apply(objects42)
        
        face30.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects43 = [NXOpen.DisplayableObject.Null] * 1 
        face31 = swept2.FindObject("FACE 10006 {(255.5310202084843,14.0049869147686,-2.3234619254777) SWEPT(5)}")
        objects43[0] = face31
        displayModification2.Apply(objects43)
        
        face31.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects44 = [NXOpen.DisplayableObject.Null] * 1 
        face32 = swept2.FindObject("FACE 10007 {(255.630886456368,12.9262259841687,-2.4043440139196) SWEPT(5)}")
        objects44[0] = face32
        displayModification2.Apply(objects44)
        
        face32.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects45 = [NXOpen.DisplayableObject.Null] * 1 
        face33 = swept2.FindObject("FACE 10008 {(255.7064538132984,12.7227332434977,-0.342466097252) SWEPT(5)}")
        objects45[0] = face33
        displayModification2.Apply(objects45)
        
        face33.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects46 = [NXOpen.DisplayableObject.Null] * 1 
        face34 = swept2.FindObject("FACE 10009 {(255.7453664429079,13.00536157471,2.0615186667913) SWEPT(5)}")
        objects46[0] = face34
        displayModification2.Apply(objects46)
        
        face34.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects47 = [NXOpen.DisplayableObject.Null] * 1 
        face35 = swept2.FindObject("FACE 10010 {(255.4076107542182,16.5452575366224,1.9589489520188) SWEPT(5)}")
        objects47[0] = face35
        displayModification2.Apply(objects47)
        
        face35.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects48 = [NXOpen.DisplayableObject.Null] * 1 
        face36 = swept2.FindObject("FACE 10011 {(254.9845153986943,20.5196147172225,0.2369040359573) SWEPT(5)}")
        objects48[0] = face36
        displayModification2.Apply(objects48)
        
        face36.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects49 = [NXOpen.DisplayableObject.Null] * 1 
        face37 = swept2.FindObject("FACE 10012 {(255.0265623540728,19.3303596132328,-2.34385283791) SWEPT(5)}")
        objects49[0] = face37
        displayModification2.Apply(objects49)
        
        face37.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects50 = [NXOpen.DisplayableObject.Null] * 1 
        face38 = swept2.FindObject("FACE 10013 {(255.2397785895237,17.0932640065222,-2.2876015897572) SWEPT(5)}")
        objects50[0] = face38
        displayModification2.Apply(objects50)
        
        face38.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects51 = [NXOpen.DisplayableObject.Null] * 1 
        face39 = swept2.FindObject("FACE 10014 {(254.9729528940538,19.6154676253028,-3.3189763675873) SWEPT(5)}")
        objects51[0] = face39
        displayModification2.Apply(objects51)
        
        face39.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects52 = [NXOpen.DisplayableObject.Null] * 1 
        face40 = swept2.FindObject("FACE 10015 {(254.908256762824,21.445331490138,0.6519410004123) SWEPT(5)}")
        objects52[0] = face40
        displayModification2.Apply(objects52)
        
        face40.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects53 = [NXOpen.DisplayableObject.Null] * 1 
        face41 = swept2.FindObject("FACE 10016 {(255.4176601487202,16.7279112669022,2.9597321843209) SWEPT(5)}")
        objects53[0] = face41
        displayModification2.Apply(objects53)
        
        face41.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects54 = [NXOpen.DisplayableObject.Null] * 1 
        face42 = swept2.FindObject("FACE 10017 {(255.8551137541262,12.220994430935,3.3623923053412) SWEPT(5)}")
        objects54[0] = face42
        displayModification2.Apply(objects54)
        
        face42.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects55 = [NXOpen.DisplayableObject.Null] * 1 
        face43 = swept2.FindObject("FACE 10018 {(256.2703475204473,7.7130456445373,2.9478969624877) SWEPT(5)}")
        objects55[0] = face43
        displayModification2.Apply(objects55)
        
        face43.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects56 = [NXOpen.DisplayableObject.Null] * 1 
        face44 = swept2.FindObject("FACE 10019 {(256.5712747518025,4.2784801251906,2.0669205926023) SWEPT(5)}")
        objects56[0] = face44
        displayModification2.Apply(objects56)
        
        face44.Color = 32767
        
        theSession.DeleteUndoMark(markId88, None)
        
        theSession.SetUndoMarkName(markId75, "扫掠")
        
        sweptBuilder2.Destroy()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression38)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression33)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression34)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression35)
        
        workPart.MeasureManager.ClearPartTransientModification()

        # ----------------------------------------------
        #   菜单：插入(S)->同步建模(Y)->偏置区域(O)...
        # ----------------------------------------------
        markId89 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        admOffsetRegionBuilder1 = workPart.Features.CreateAdmOffsetRegionBuilder(NXOpen.Features.AdmOffsetRegion.Null)
        
        admOffsetRegionBuilder1.FaceToOffset.RelationScope = 1023
        
        admOffsetRegionBuilder1.FaceToOffset.CoplanarEnabled = False
        
        admOffsetRegionBuilder1.FaceToOffset.CoplanarAxesEnabled = False
        
        admOffsetRegionBuilder1.FaceToOffset.CoaxialEnabled = False
        
        admOffsetRegionBuilder1.FaceToOffset.SameOrbitEnabled = False
        
        admOffsetRegionBuilder1.FaceToOffset.EqualDiameterEnabled = False
        
        admOffsetRegionBuilder1.FaceToOffset.TangentEnabled = False
        
        admOffsetRegionBuilder1.FaceToOffset.SymmetricEnabled = False
        
        admOffsetRegionBuilder1.FaceToOffset.OffsetEnabled = False
        
        admOffsetRegionBuilder1.FaceToOffset.UseFaceBrowse = True
        
        admOffsetRegionBuilder1.Distance.SetFormula("0.001")
        
        admOffsetRegionBuilder1.Distance.SetFormula("0")
        
        theSession.SetUndoMarkName(markId89, "偏置区域 对话框")
        
        scCollector1 = workPart.FindObject("ENTITY 113 2")
        rules10 = []
        scCollector1.ReplaceRules(rules10, False)
        
        selectionIntentRuleOptions10 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions10.SetSelectedFromInactive(False)
        
        features6 = [NXOpen.Features.Feature.Null] * 1 
        features6[0] = swept2
        faceFeatureRule1 = workPart.ScRuleFactory.CreateRuleFaceFeature(features6, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions10)
        
        selectionIntentRuleOptions10.Dispose()
        rules11 = [None] * 1 
        rules11[0] = faceFeatureRule1
        admOffsetRegionBuilder1.FaceToOffset.FaceCollector.ReplaceRules(rules11, False)
        
        admOffsetRegionBuilder1.Distance.SetFormula(str(bias))
        
        admOffsetRegionBuilder1.ReverseDirection = True
        
        admOffsetRegionBuilder1.ReverseDirection = False
        
        markId90 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "偏置区域")
        
        theSession.DeleteUndoMark(markId90, None)
        
        markId91 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "偏置区域")
        
        nXObject13 = admOffsetRegionBuilder1.Commit()
        
        theSession.DeleteUndoMark(markId91, None)
        
        theSession.SetUndoMarkName(markId89, "偏置区域")
        
        expression39 = admOffsetRegionBuilder1.Distance
        admOffsetRegionBuilder1.Destroy()
        
        # ----------------------------------------------
        #   菜单：插入(S)->组合(B)->合并(U)...
        # ----------------------------------------------
        markId92 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        booleanBuilder1 = workPart.Features.CreateBooleanBuilderUsingCollector(NXOpen.Features.BooleanFeature.Null)
        
        scCollector2 = booleanBuilder1.ToolBodyCollector
        
        scCollector3 = booleanBuilder1.TargetBodyCollector
        
        booleanRegionSelect1 = booleanBuilder1.BooleanRegionSelect
        
        booleanBuilder1.Tolerance = 0.001
        
        booleanBuilder1.CopyTargets = True
        
        booleanBuilder1.CopyTools = True
        
        scCollector4 = booleanBuilder1.TargetBodyCollector
        
        booleanBuilder1.Operation = NXOpen.Features.Feature.BooleanType.Unite
        
        theSession.SetUndoMarkName(markId92, "合并 对话框")
        
        scCollector5 = workPart.ScCollectors.CreateCollector()
        
        selectionIntentRuleOptions11 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions11.SetSelectedFromInactive(False)
        
        bodies1 = [NXOpen.Body.Null] * 1 
        bodies1[0] = body1
        bodyDumbRule1 = workPart.ScRuleFactory.CreateRuleBodyDumb(bodies1, True, selectionIntentRuleOptions11)
        
        selectionIntentRuleOptions11.Dispose()
        rules12 = [None] * 1 
        rules12[0] = bodyDumbRule1
        scCollector5.ReplaceRules(rules12, False)
        
        booleanBuilder1.TargetBodyCollector = scCollector5
        
        targets1 = [NXOpen.TaggedObject.Null] * 1 
        targets1[0] = body1
        booleanRegionSelect1.AssignTargets(targets1)
        
        scCollector6 = workPart.ScCollectors.CreateCollector()
        
        selectionIntentRuleOptions12 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions12.SetSelectedFromInactive(False)
        
        features7 = [NXOpen.Features.Feature.Null] * 1 
        admOffsetRegion1 = nXObject13
        features7[0] = admOffsetRegion1
        bodyFeatureRule1 = workPart.ScRuleFactory.CreateRuleBodyFeature(features7, False, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions12)
        
        selectionIntentRuleOptions12.Dispose()
        rules13 = [None] * 1 
        rules13[0] = bodyFeatureRule1
        scCollector6.ReplaceRules(rules13, False)
        
        booleanBuilder1.ToolBodyCollector = scCollector6
        
        targets2 = [NXOpen.TaggedObject.Null] * 1 
        targets2[0] = body1
        booleanRegionSelect1.AssignTargets(targets2)
        
        markId93 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "合并")
        
        theSession.DeleteUndoMark(markId93, None)
        
        markId94 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "合并")
        
        nXObject14 = booleanBuilder1.Commit()
        
        theSession.DeleteUndoMark(markId94, None)
        
        theSession.SetUndoMarkName(markId92, "合并")
        
        booleanBuilder1.Destroy()
        
        markId95 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects57 = [NXOpen.DisplayableObject.Null] * 1 
        body2 = workPart.Bodies.FindObject("SWEPT(5)")
        objects57[0] = body2
        theSession.DisplayManager.BlankObjects(objects57)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        markId96 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Show")
        
        objects58 = [NXOpen.DisplayableObject.Null] * 1 
        objects58[0] = body2
        theSession.DisplayManager.ShowObjects(objects58, NXOpen.DisplayManager.LayerSetting.ChangeLayerToSelectable)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.ShowOnly)
        
        markId97 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects59 = [NXOpen.DisplayableObject.Null] * 1 
        body3 = workPart.Bodies.FindObject("UNITE(7)")
        objects59[0] = body3
        theSession.DisplayManager.BlankObjects(objects59)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        markId98 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects60 = [NXOpen.DisplayableObject.Null] * 1 
        objects60[0] = body2
        theSession.DisplayManager.BlankObjects(objects60)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：分析(L)  ->测量(S)...
        # ----------------------------------------------
        markId99 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        theSession.SetUndoMarkName(markId99, "测量 对话框")
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector7 = workPart.ScCollectors.CreateCollector()
        
        scCollector7.SetMultiComponent()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        selectionIntentRuleOptions13 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions13.SetSelectedFromInactive(False)
        
        features8 = [NXOpen.Features.Feature.Null] * 1 
        features8[0] = swept1
        bodyFeatureRule2 = workPart.ScRuleFactory.CreateRuleBodyFeature(features8, True, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions13)
        
        selectionIntentRuleOptions13.Dispose()
        rules14 = [None] * 1 
        rules14[0] = bodyFeatureRule2
        scCollector7.ReplaceRules(rules14, False)
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector8 = workPart.ScCollectors.CreateCollector()
        
        scCollector8.SetMultiComponent()
        
        markId100 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        theSession.DeleteUndoMark(markId100, None)
        
        markId101 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        markId102 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Measurement Apply")
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        measureMaster1 = workPart.MeasureManager.MasterMeasurement()
        
        measureMaster1.SequenceType = NXOpen.MeasureMaster.Sequence.Free
        
        measureMaster1.UpdateAtTimestamp = True
        
        measureMaster1.SetNameSuffix("实体")
        
        massUnits1 = [NXOpen.Unit.Null] * 8 
        massUnits1[0] = unit1
        unit3 = workPart.UnitCollection.FindObject("SquareMilliMeter")
        massUnits1[1] = unit3
        unit4 = workPart.UnitCollection.FindObject("CubicMilliMeter")
        massUnits1[2] = unit4
        unit5 = workPart.UnitCollection.FindObject("KilogramPerCubicMilliMeter")
        massUnits1[3] = unit5
        unit6 = workPart.UnitCollection.FindObject("Kilogram")
        massUnits1[4] = unit6
        unit7 = workPart.UnitCollection.FindObject("KilogramMilliMeterSquared")
        massUnits1[5] = unit7
        unit8 = workPart.UnitCollection.FindObject("KilogramMilliMeter")
        massUnits1[6] = unit8
        unit9 = workPart.UnitCollection.FindObject("Newton")
        massUnits1[7] = unit9
        measureElement1 = workPart.MeasureManager.BodyElement(measureMaster1, massUnits1, 0.98999999999999999, scCollector7)
        
        measureElement1.MeasureObject1 = NXOpen.MeasureElement.Measure.Object
        
        measureElement1.SingleSelect1 = True
        
        measureElement1.SetExpressionState(0, False)
        
        measureElement1.SetGeometryState(0, False)
        
        measureElement1.SetAnnotationState(0, False)
        
        measureElement1.SetApproximateState(0, False)
        
        measureElement1.SetExpressionState(1, True)
        
        measureElement1.SetGeometryState(1, False)
        
        measureElement1.SetAnnotationState(1, True)
        
        measureElement1.SetApproximateState(1, False)
        
        measureElement1.SetExpressionState(2, False)
        
        measureElement1.SetGeometryState(2, False)
        
        measureElement1.SetAnnotationState(2, False)
        
        measureElement1.SetApproximateState(2, False)
        
        measureElement1.SetExpressionState(3, False)
        
        measureElement1.SetGeometryState(3, False)
        
        measureElement1.SetAnnotationState(3, False)
        
        measureElement1.SetApproximateState(3, False)
        
        measureElement1.SetExpressionState(4, False)
        
        measureElement1.SetGeometryState(4, False)
        
        measureElement1.SetAnnotationState(4, False)
        
        measureElement1.SetApproximateState(4, False)
        
        measureElement1.SetExpressionState(5, False)
        
        measureElement1.SetGeometryState(5, False)
        
        measureElement1.SetAnnotationState(5, False)
        
        measureElement1.SetApproximateState(5, False)
        
        measureElement1.SetExpressionState(6, False)
        
        measureElement1.SetGeometryState(6, False)
        
        measureElement1.SetAnnotationState(6, False)
        
        measureElement1.SetApproximateState(6, False)
        
        measureElement1.SetExpressionState(7, False)
        
        measureElement1.SetGeometryState(7, False)
        
        measureElement1.SetAnnotationState(7, False)
        
        measureElement1.SetApproximateState(7, False)
        
        measureElement1.SetExpressionState(8, False)
        
        measureElement1.SetGeometryState(8, False)
        
        measureElement1.SetAnnotationState(8, False)
        
        measureElement1.SetApproximateState(8, False)
        
        measureElement1.SetExpressionState(9, False)
        
        measureElement1.SetGeometryState(9, False)
        
        measureElement1.SetAnnotationState(9, False)
        
        measureElement1.SetApproximateState(9, False)
        
        measureElement1.SetExpressionState(10, False)
        
        measureElement1.SetGeometryState(10, False)
        
        measureElement1.SetAnnotationState(10, False)
        
        measureElement1.SetApproximateState(10, False)
        
        measureElement1.SetExpressionState(11, False)
        
        measureElement1.SetGeometryState(11, False)
        
        measureElement1.SetAnnotationState(11, False)
        
        measureElement1.SetApproximateState(11, False)
        
        measureElement1.SetExpressionState(12, False)
        
        measureElement1.SetGeometryState(12, False)
        
        measureElement1.SetAnnotationState(12, False)
        
        measureElement1.SetApproximateState(12, False)
        
        measureElement1.SetExpressionState(13, False)
        
        measureElement1.SetGeometryState(13, False)
        
        measureElement1.SetAnnotationState(13, False)
        
        measureElement1.SetApproximateState(13, False)
        
        measureElement1.SetExpressionState(14, False)
        
        measureElement1.SetGeometryState(14, False)
        
        measureElement1.SetAnnotationState(14, False)
        
        measureElement1.SetApproximateState(14, False)
        
        measureElement1.SetExpressionState(15, False)
        
        measureElement1.SetGeometryState(15, False)
        
        measureElement1.SetAnnotationState(15, False)
        
        measureElement1.SetApproximateState(15, False)
        
        measureElement1.SetExpressionState(16, False)
        
        measureElement1.SetGeometryState(16, False)
        
        measureElement1.SetAnnotationState(16, False)
        
        measureElement1.SetApproximateState(16, False)
        
        measureElement1.SetExpressionState(17, False)
        
        measureElement1.SetGeometryState(17, False)
        
        measureElement1.SetAnnotationState(17, False)
        
        measureElement1.SetApproximateState(17, False)
        
        measureElement1.SetExpressionState(18, False)
        
        measureElement1.SetGeometryState(18, False)
        
        measureElement1.SetAnnotationState(18, False)
        
        measureElement1.SetApproximateState(18, False)
        
        measureElement1.SetExpressionState(19, False)
        
        measureElement1.SetGeometryState(19, False)
        
        measureElement1.SetAnnotationState(19, False)
        
        measureElement1.SetApproximateState(19, False)
        
        measureElement1.SetExpressionState(20, False)
        
        measureElement1.SetGeometryState(20, False)
        
        measureElement1.SetAnnotationState(20, False)
        
        measureElement1.SetApproximateState(20, False)
        
        measureElement1.SetExpressionState(21, False)
        
        measureElement1.SetGeometryState(21, False)
        
        measureElement1.SetAnnotationState(21, False)
        
        measureElement1.SetApproximateState(21, False)
        
        measureElement1.SetExpressionState(22, False)
        
        measureElement1.SetGeometryState(22, False)
        
        measureElement1.SetAnnotationState(22, False)
        
        measureElement1.SetApproximateState(22, False)
        
        measureElement1.SetExpressionState(23, False)
        
        measureElement1.SetGeometryState(23, False)
        
        measureElement1.SetAnnotationState(23, False)
        
        measureElement1.SetApproximateState(23, False)
        
        measureElement1.SetExpressionState(24, False)
        
        measureElement1.SetGeometryState(24, False)
        
        measureElement1.SetAnnotationState(24, False)
        
        measureElement1.SetApproximateState(24, False)
        
        measureElement1.SetExpressionState(25, False)
        
        measureElement1.SetGeometryState(25, False)
        
        measureElement1.SetAnnotationState(25, False)
        
        measureElement1.SetApproximateState(25, False)
        
        measureElement2 = measureMaster1.GetMeasureElement(0)
        
        measureElement2.CreateGeometry()
        
        measureElement3 = measureMaster1.GetMeasureElement(0)
        
        annotation1 = measureElement3.CreateAnnotation()
        
        measureElement4 = measureMaster1.GetMeasureElement(0)
        
        measureElement5 = measureMaster1.GetMeasureElement(0)
        
        measureElement5.EditAnnotation()
        
        measureMaster1.FixupModelingParents()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        markId103 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Measurement Update")
        
        nErrs10 = theSession.UpdateManager.DoUpdate(markId103)
        
        theSession.DeleteUndoMark(markId103, "Measurement Update")
        
        theSession.DeleteUndoMark(markId102, "Measurement Apply")
        
        datadeleted1 = theSession.DeleteTransientDynamicSectionCutData()
        
        theSession.DeleteUndoMark(markId101, None)
        
        theSession.SetUndoMarkName(markId99, "测量")
        
        scCollector8.Destroy()
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        # ----------------------------------------------
        #   菜单：分析(L)  ->测量(S)...
        # ----------------------------------------------
        markId104 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        theSession.SetUndoMarkName(markId104, "测量 对话框")
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector9 = workPart.ScCollectors.CreateCollector()
        
        scCollector9.SetMultiComponent()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        selectionIntentRuleOptions14 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions14.SetSelectedFromInactive(False)
        
        features9 = [NXOpen.Features.Feature.Null] * 1 
        features9[0] = admOffsetRegion1
        bodyFeatureRule3 = workPart.ScRuleFactory.CreateRuleBodyFeature(features9, True, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions14)
        
        selectionIntentRuleOptions14.Dispose()
        rules15 = [None] * 1 
        rules15[0] = bodyFeatureRule3
        scCollector9.ReplaceRules(rules15, False)
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector10 = workPart.ScCollectors.CreateCollector()
        
        scCollector10.SetMultiComponent()
        
        markId105 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        theSession.DeleteUndoMark(markId105, None)
        
        markId106 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        markId107 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Measurement Apply")
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        measureMaster2 = workPart.MeasureManager.MasterMeasurement()
        
        measureMaster2.SequenceType = NXOpen.MeasureMaster.Sequence.Free
        
        measureMaster2.UpdateAtTimestamp = True
        
        measureMaster2.SetNameSuffix("实体")
        
        massUnits2 = [NXOpen.Unit.Null] * 8 
        massUnits2[0] = unit1
        massUnits2[1] = unit3
        massUnits2[2] = unit4
        massUnits2[3] = unit5
        massUnits2[4] = unit6
        massUnits2[5] = unit7
        massUnits2[6] = unit8
        massUnits2[7] = unit9
        measureElement6 = workPart.MeasureManager.BodyElement(measureMaster2, massUnits2, 0.98999999999999999, scCollector9)
        
        measureElement6.MeasureObject1 = NXOpen.MeasureElement.Measure.Object
        
        measureElement6.SingleSelect1 = True
        
        measureElement6.SetExpressionState(0, False)
        
        measureElement6.SetGeometryState(0, False)
        
        measureElement6.SetAnnotationState(0, False)
        
        measureElement6.SetApproximateState(0, False)
        
        measureElement6.SetExpressionState(1, True)
        
        measureElement6.SetGeometryState(1, False)
        
        measureElement6.SetAnnotationState(1, True)
        
        measureElement6.SetApproximateState(1, False)
        
        measureElement6.SetExpressionState(2, False)
        
        measureElement6.SetGeometryState(2, False)
        
        measureElement6.SetAnnotationState(2, False)
        
        measureElement6.SetApproximateState(2, False)
        
        measureElement6.SetExpressionState(3, False)
        
        measureElement6.SetGeometryState(3, False)
        
        measureElement6.SetAnnotationState(3, False)
        
        measureElement6.SetApproximateState(3, False)
        
        measureElement6.SetExpressionState(4, False)
        
        measureElement6.SetGeometryState(4, False)
        
        measureElement6.SetAnnotationState(4, False)
        
        measureElement6.SetApproximateState(4, False)
        
        measureElement6.SetExpressionState(5, False)
        
        measureElement6.SetGeometryState(5, False)
        
        measureElement6.SetAnnotationState(5, False)
        
        measureElement6.SetApproximateState(5, False)
        
        measureElement6.SetExpressionState(6, False)
        
        measureElement6.SetGeometryState(6, False)
        
        measureElement6.SetAnnotationState(6, False)
        
        measureElement6.SetApproximateState(6, False)
        
        measureElement6.SetExpressionState(7, False)
        
        measureElement6.SetGeometryState(7, False)
        
        measureElement6.SetAnnotationState(7, False)
        
        measureElement6.SetApproximateState(7, False)
        
        measureElement6.SetExpressionState(8, False)
        
        measureElement6.SetGeometryState(8, False)
        
        measureElement6.SetAnnotationState(8, False)
        
        measureElement6.SetApproximateState(8, False)
        
        measureElement6.SetExpressionState(9, False)
        
        measureElement6.SetGeometryState(9, False)
        
        measureElement6.SetAnnotationState(9, False)
        
        measureElement6.SetApproximateState(9, False)
        
        measureElement6.SetExpressionState(10, False)
        
        measureElement6.SetGeometryState(10, False)
        
        measureElement6.SetAnnotationState(10, False)
        
        measureElement6.SetApproximateState(10, False)
        
        measureElement6.SetExpressionState(11, False)
        
        measureElement6.SetGeometryState(11, False)
        
        measureElement6.SetAnnotationState(11, False)
        
        measureElement6.SetApproximateState(11, False)
        
        measureElement6.SetExpressionState(12, False)
        
        measureElement6.SetGeometryState(12, False)
        
        measureElement6.SetAnnotationState(12, False)
        
        measureElement6.SetApproximateState(12, False)
        
        measureElement6.SetExpressionState(13, False)
        
        measureElement6.SetGeometryState(13, False)
        
        measureElement6.SetAnnotationState(13, False)
        
        measureElement6.SetApproximateState(13, False)
        
        measureElement6.SetExpressionState(14, False)
        
        measureElement6.SetGeometryState(14, False)
        
        measureElement6.SetAnnotationState(14, False)
        
        measureElement6.SetApproximateState(14, False)
        
        measureElement6.SetExpressionState(15, False)
        
        measureElement6.SetGeometryState(15, False)
        
        measureElement6.SetAnnotationState(15, False)
        
        measureElement6.SetApproximateState(15, False)
        
        measureElement6.SetExpressionState(16, False)
        
        measureElement6.SetGeometryState(16, False)
        
        measureElement6.SetAnnotationState(16, False)
        
        measureElement6.SetApproximateState(16, False)
        
        measureElement6.SetExpressionState(17, False)
        
        measureElement6.SetGeometryState(17, False)
        
        measureElement6.SetAnnotationState(17, False)
        
        measureElement6.SetApproximateState(17, False)
        
        measureElement6.SetExpressionState(18, False)
        
        measureElement6.SetGeometryState(18, False)
        
        measureElement6.SetAnnotationState(18, False)
        
        measureElement6.SetApproximateState(18, False)
        
        measureElement6.SetExpressionState(19, False)
        
        measureElement6.SetGeometryState(19, False)
        
        measureElement6.SetAnnotationState(19, False)
        
        measureElement6.SetApproximateState(19, False)
        
        measureElement6.SetExpressionState(20, False)
        
        measureElement6.SetGeometryState(20, False)
        
        measureElement6.SetAnnotationState(20, False)
        
        measureElement6.SetApproximateState(20, False)
        
        measureElement6.SetExpressionState(21, False)
        
        measureElement6.SetGeometryState(21, False)
        
        measureElement6.SetAnnotationState(21, False)
        
        measureElement6.SetApproximateState(21, False)
        
        measureElement6.SetExpressionState(22, False)
        
        measureElement6.SetGeometryState(22, False)
        
        measureElement6.SetAnnotationState(22, False)
        
        measureElement6.SetApproximateState(22, False)
        
        measureElement6.SetExpressionState(23, False)
        
        measureElement6.SetGeometryState(23, False)
        
        measureElement6.SetAnnotationState(23, False)
        
        measureElement6.SetApproximateState(23, False)
        
        measureElement6.SetExpressionState(24, False)
        
        measureElement6.SetGeometryState(24, False)
        
        measureElement6.SetAnnotationState(24, False)
        
        measureElement6.SetApproximateState(24, False)
        
        measureElement6.SetExpressionState(25, False)
        
        measureElement6.SetGeometryState(25, False)
        
        measureElement6.SetAnnotationState(25, False)
        
        measureElement6.SetApproximateState(25, False)
        
        measureElement7 = measureMaster2.GetMeasureElement(0)
        
        measureElement7.CreateGeometry()
        
        measureElement8 = measureMaster2.GetMeasureElement(0)
        
        annotation2 = measureElement8.CreateAnnotation()
        
        measureElement9 = measureMaster2.GetMeasureElement(0)
        
        measureElement10 = measureMaster2.GetMeasureElement(0)
        
        measureElement10.EditAnnotation()
        
        measureMaster2.FixupModelingParents()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        markId108 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Measurement Update")
        
        nErrs11 = theSession.UpdateManager.DoUpdate(markId108)
        
        theSession.DeleteUndoMark(markId108, "Measurement Update")
        
        theSession.DeleteUndoMark(markId107, "Measurement Apply")
        
        datadeleted2 = theSession.DeleteTransientDynamicSectionCutData()
        
        theSession.DeleteUndoMark(markId106, None)
        
        theSession.SetUndoMarkName(markId104, "测量")
        
        scCollector10.Destroy()
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        # ----------------------------------------------
        #   菜单：分析(L)  ->测量(S)...
        # ----------------------------------------------
        markId109 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        theSession.SetUndoMarkName(markId109, "测量 对话框")
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector11 = workPart.ScCollectors.CreateCollector()
        
        scCollector11.SetMultiComponent()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        selectionIntentRuleOptions15 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions15.SetSelectedFromInactive(False)
        
        features10 = [NXOpen.Features.Feature.Null] * 1 
        booleanFeature1 = nXObject14
        features10[0] = booleanFeature1
        bodyFeatureRule4 = workPart.ScRuleFactory.CreateRuleBodyFeature(features10, True, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions15)
        
        selectionIntentRuleOptions15.Dispose()
        rules16 = [None] * 1 
        rules16[0] = bodyFeatureRule4
        scCollector11.ReplaceRules(rules16, False)
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector12 = workPart.ScCollectors.CreateCollector()
        
        scCollector12.SetMultiComponent()
        
        markId110 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        theSession.DeleteUndoMark(markId110, None)
        
        markId111 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        markId112 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Measurement Apply")
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        measureMaster3 = workPart.MeasureManager.MasterMeasurement()
        
        measureMaster3.SequenceType = NXOpen.MeasureMaster.Sequence.Free
        
        measureMaster3.UpdateAtTimestamp = True
        
        measureMaster3.SetNameSuffix("实体")
        
        massUnits3 = [NXOpen.Unit.Null] * 8 
        massUnits3[0] = unit1
        massUnits3[1] = unit3
        massUnits3[2] = unit4
        massUnits3[3] = unit5
        massUnits3[4] = unit6
        massUnits3[5] = unit7
        massUnits3[6] = unit8
        massUnits3[7] = unit9
        measureElement11 = workPart.MeasureManager.BodyElement(measureMaster3, massUnits3, 0.98999999999999999, scCollector11)
        
        measureElement11.MeasureObject1 = NXOpen.MeasureElement.Measure.Object
        
        measureElement11.SingleSelect1 = True
        
        measureElement11.SetExpressionState(0, False)
        
        measureElement11.SetGeometryState(0, False)
        
        measureElement11.SetAnnotationState(0, False)
        
        measureElement11.SetApproximateState(0, False)
        
        measureElement11.SetExpressionState(1, True)
        
        measureElement11.SetGeometryState(1, False)
        
        measureElement11.SetAnnotationState(1, True)
        
        measureElement11.SetApproximateState(1, False)
        
        measureElement11.SetExpressionState(2, False)
        
        measureElement11.SetGeometryState(2, False)
        
        measureElement11.SetAnnotationState(2, False)
        
        measureElement11.SetApproximateState(2, False)
        
        measureElement11.SetExpressionState(3, False)
        
        measureElement11.SetGeometryState(3, False)
        
        measureElement11.SetAnnotationState(3, False)
        
        measureElement11.SetApproximateState(3, False)
        
        measureElement11.SetExpressionState(4, False)
        
        measureElement11.SetGeometryState(4, False)
        
        measureElement11.SetAnnotationState(4, False)
        
        measureElement11.SetApproximateState(4, False)
        
        measureElement11.SetExpressionState(5, False)
        
        measureElement11.SetGeometryState(5, False)
        
        measureElement11.SetAnnotationState(5, False)
        
        measureElement11.SetApproximateState(5, False)
        
        measureElement11.SetExpressionState(6, False)
        
        measureElement11.SetGeometryState(6, False)
        
        measureElement11.SetAnnotationState(6, False)
        
        measureElement11.SetApproximateState(6, False)
        
        measureElement11.SetExpressionState(7, False)
        
        measureElement11.SetGeometryState(7, False)
        
        measureElement11.SetAnnotationState(7, False)
        
        measureElement11.SetApproximateState(7, False)
        
        measureElement11.SetExpressionState(8, False)
        
        measureElement11.SetGeometryState(8, False)
        
        measureElement11.SetAnnotationState(8, False)
        
        measureElement11.SetApproximateState(8, False)
        
        measureElement11.SetExpressionState(9, False)
        
        measureElement11.SetGeometryState(9, False)
        
        measureElement11.SetAnnotationState(9, False)
        
        measureElement11.SetApproximateState(9, False)
        
        measureElement11.SetExpressionState(10, False)
        
        measureElement11.SetGeometryState(10, False)
        
        measureElement11.SetAnnotationState(10, False)
        
        measureElement11.SetApproximateState(10, False)
        
        measureElement11.SetExpressionState(11, False)
        
        measureElement11.SetGeometryState(11, False)
        
        measureElement11.SetAnnotationState(11, False)
        
        measureElement11.SetApproximateState(11, False)
        
        measureElement11.SetExpressionState(12, False)
        
        measureElement11.SetGeometryState(12, False)
        
        measureElement11.SetAnnotationState(12, False)
        
        measureElement11.SetApproximateState(12, False)
        
        measureElement11.SetExpressionState(13, False)
        
        measureElement11.SetGeometryState(13, False)
        
        measureElement11.SetAnnotationState(13, False)
        
        measureElement11.SetApproximateState(13, False)
        
        measureElement11.SetExpressionState(14, False)
        
        measureElement11.SetGeometryState(14, False)
        
        measureElement11.SetAnnotationState(14, False)
        
        measureElement11.SetApproximateState(14, False)
        
        measureElement11.SetExpressionState(15, False)
        
        measureElement11.SetGeometryState(15, False)
        
        measureElement11.SetAnnotationState(15, False)
        
        measureElement11.SetApproximateState(15, False)
        
        measureElement11.SetExpressionState(16, False)
        
        measureElement11.SetGeometryState(16, False)
        
        measureElement11.SetAnnotationState(16, False)
        
        measureElement11.SetApproximateState(16, False)
        
        measureElement11.SetExpressionState(17, False)
        
        measureElement11.SetGeometryState(17, False)
        
        measureElement11.SetAnnotationState(17, False)
        
        measureElement11.SetApproximateState(17, False)
        
        measureElement11.SetExpressionState(18, False)
        
        measureElement11.SetGeometryState(18, False)
        
        measureElement11.SetAnnotationState(18, False)
        
        measureElement11.SetApproximateState(18, False)
        
        measureElement11.SetExpressionState(19, False)
        
        measureElement11.SetGeometryState(19, False)
        
        measureElement11.SetAnnotationState(19, False)
        
        measureElement11.SetApproximateState(19, False)
        
        measureElement11.SetExpressionState(20, False)
        
        measureElement11.SetGeometryState(20, False)
        
        measureElement11.SetAnnotationState(20, False)
        
        measureElement11.SetApproximateState(20, False)
        
        measureElement11.SetExpressionState(21, False)
        
        measureElement11.SetGeometryState(21, False)
        
        measureElement11.SetAnnotationState(21, False)
        
        measureElement11.SetApproximateState(21, False)
        
        measureElement11.SetExpressionState(22, False)
        
        measureElement11.SetGeometryState(22, False)
        
        measureElement11.SetAnnotationState(22, False)
        
        measureElement11.SetApproximateState(22, False)
        
        measureElement11.SetExpressionState(23, False)
        
        measureElement11.SetGeometryState(23, False)
        
        measureElement11.SetAnnotationState(23, False)
        
        measureElement11.SetApproximateState(23, False)
        
        measureElement11.SetExpressionState(24, False)
        
        measureElement11.SetGeometryState(24, False)
        
        measureElement11.SetAnnotationState(24, False)
        
        measureElement11.SetApproximateState(24, False)
        
        measureElement11.SetExpressionState(25, False)
        
        measureElement11.SetGeometryState(25, False)
        
        measureElement11.SetAnnotationState(25, False)
        
        measureElement11.SetApproximateState(25, False)
        
        measureElement12 = measureMaster3.GetMeasureElement(0)
        
        measureElement12.CreateGeometry()
        
        measureElement13 = measureMaster3.GetMeasureElement(0)
        
        annotation3 = measureElement13.CreateAnnotation()
        
        measureElement14 = measureMaster3.GetMeasureElement(0)
        
        measureElement15 = measureMaster3.GetMeasureElement(0)
        
        measureElement15.EditAnnotation()
        
        measureMaster3.FixupModelingParents()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        markId113 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Measurement Update")
        
        nErrs12 = theSession.UpdateManager.DoUpdate(markId113)
        
        theSession.DeleteUndoMark(markId113, "Measurement Update")
        
        theSession.DeleteUndoMark(markId112, "Measurement Apply")
        
        datadeleted3 = theSession.DeleteTransientDynamicSectionCutData()
        
        theSession.DeleteUndoMark(markId111, None)
        
        theSession.SetUndoMarkName(markId109, "测量")
        
        scCollector12.Destroy()
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        markId114 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects61 = [NXOpen.DisplayableObject.Null] * 1 
        generalLabel1 = annotation1
        objects61[0] = generalLabel1
        theSession.DisplayManager.BlankObjects(objects61)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        markId115 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects62 = [NXOpen.DisplayableObject.Null] * 1 
        generalLabel2 = annotation2
        objects62[0] = generalLabel2
        theSession.DisplayManager.BlankObjects(objects62)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        markId116 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects63 = [NXOpen.DisplayableObject.Null] * 1 
        generalLabel3 = annotation3
        objects63[0] = generalLabel3

        springback_strip_volume = float(generalLabel1.GetText()[0].split()[1])
        print(springback_strip_volume)

        prediction_strip_volume = float(generalLabel2.GetText()[0].split()[1])
        print(prediction_strip_volume)

        union_strip_volume = float(generalLabel3.GetText()[0].split()[1])
        print(union_strip_volume)

        intersection_strip_volume = springback_strip_volume + prediction_strip_volume - union_strip_volume
        print(intersection_strip_volume)

        iou_3d = intersection_strip_volume / union_strip_volume
        print(iou_3d)
        theSession.DisplayManager.BlankObjects(objects63)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：文件(F)->保存(S)
        # ----------------------------------------------
        partSaveStatus1 = workPart.Save(NXOpen.BasePart.SaveComponents.TrueValue, NXOpen.BasePart.CloseAfterSave.FalseValue)
        
        partSaveStatus1.Dispose()
        partCloseResponses1 = theSession.Parts.NewPartCloseResponses()
        
        workPart.Close(NXOpen.BasePart.CloseWholeTree.FalseValue, NXOpen.BasePart.CloseModified.UseResponses, partCloseResponses1)
        
        workPart = NXOpen.Part.Null
        displayPart = NXOpen.Part.Null
        partCloseResponses1.Dispose()
        theSession.ApplicationSwitchImmediate("UG_APP_NOPART")
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->3 停止操作记录录制
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：工具(T)->操作记录(J)->停止录制(S)
        # ----------------------------------------------
        return iou_3d
    except:
        workPart.Close(NXOpen.BasePart.CloseWholeTree.FalseValue, NXOpen.BasePart.CloseModified.UseResponses, None)
    
        workPart = NXOpen.Part.Null
        displayPart = NXOpen.Part.Null
        theSession.ApplicationSwitchImmediate("UG_APP_NOPART")
        return None
    

if __name__ == '__main__':
    iou_3d = 0.0
    cnt = 0
    error = []
    for i in range(541, 600 + 1):
        cnt += 1
        data_root = "G:\\PHD2\\Code\\e2e_ablation\\data"
        strip_section_stp_path = f"{data_root}\\strip_section_stp\\type_4\\strip_section_{i:0>4}.stp"
        iou_3d_prt_path = f"{data_root}\\iou_3d_prt\\type_4\\iou_3d_{i:0>4}.prt"
        iou_3d_txt_path = f"{data_root}\\iou_3d_txt\\type_4\\iou_3d_{i:0>4}.txt"
        springback_strip_line_path = f"{data_root}\\springback_strip_line\\type_4\\springback_strip_line_{i:0>4}.txt"
        prediction_line_path = f"{data_root}\\prediction_line\\type_4\\{i:0>4}.txt"
        sys.stdout = open(iou_3d_txt_path, 'w', encoding = "utf-8")
        no_bias_return = without_bias(strip_section_stp_path, iou_3d_prt_path, springback_strip_line_path, prediction_line_path)
        if no_bias_return is not None:
            iou_3d += no_bias_return
        else:
            bias = [0.01, 0.05, 0.08, 0.1, 0.12, 0.15]
            flag = False
            for b in bias:
                bias_return = with_bias(strip_section_stp_path, iou_3d_prt_path, springback_strip_line_path, prediction_line_path, b)
                if bias_return is not None:
                    iou_3d += bias_return
                    flag = True
                    break
            if not flag:
                error.append(i)
                cnt -= 1
        sys.stdout = sys.__stdout__
    sys.stdout = open(f"{data_root}\\iou_3d_txt\\type_4\\iou_3d_total.txt", 'w', encoding = "utf-8")
    print(iou_3d / cnt)
    print(cnt)
    print(error)
    sys.stdout = sys.__stdout__