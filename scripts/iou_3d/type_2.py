# NX 1953
# Journal created by Joker on Tue Jan 23 00:06:25 2024 中国标准时间

#
import sys
import math
import NXOpen
import NXOpen.Annotations
import NXOpen.Features
import NXOpen.GeometricUtilities
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
        datumCsys1 = workPart.Features.FindObject("DATUM_CSYS(0)")
        point1 = datumCsys1.FindObject("POINT 1")
        objects1[1] = point1
        cartesianCoordinateSystem1 = datumCsys1.FindObject("CSYSTEM 1")
        objects1[2] = cartesianCoordinateSystem1
        datumPlane1 = workPart.Datums.FindObject("DATUM_CSYS(0) YZ plane")
        objects1[3] = datumPlane1
        datumAxis2 = workPart.Datums.FindObject("DATUM_CSYS(0) Z axis")
        objects1[4] = datumAxis2
        datumAxis3 = workPart.Datums.FindObject("DATUM_CSYS(0) X axis")
        objects1[5] = datumAxis3
        datumPlane2 = workPart.Datums.FindObject("DATUM_CSYS(0) XZ plane")
        objects1[6] = datumPlane2
        datumPlane3 = workPart.Datums.FindObject("DATUM_CSYS(0) XY plane")
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
        
        step214Importer1.OutputFile = "G:\\PHD2\\Code\\e2e_ablation\\scripts\\iou_3d\\iou_3d_strip_section_type_2.prt"
        
        step214Importer1.SettingsFile = "D:\\NX1953\\NX1953\\step214ug\\step214ug.def"
        
        step214Importer1.ObjectTypes.ProductData = True
        
        step214Importer1.OutputFile = ""
        
        theSession.SetUndoMarkName(markId5, "导入 STEP214 文件 对话框")
        
        step214Importer1.SetMode(NXOpen.BaseImporter.Mode.NativeFileSystem)
        
        step214Importer1.OutputFile = "G:\\PHD2\\Code\\e2e_ablation\\scripts\\iou_3d\\iou_3d_strip_section_type_2.prt"
        
        step214Importer1.InputFile = strip_section_stp_path
        
        step214Importer1.SetMode(NXOpen.BaseImporter.Mode.NativeFileSystem)
        
        step214Importer1.InputFile = strip_section_stp_path
        
        markId6 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "导入 STEP214 文件")
        
        step214Importer1.FileOpenFlag = False
        
        step214Importer1.ProcessHoldFlag = True
        
        nXObject2 = step214Importer1.Commit()
        
        theSession.DeleteUndoMark(markId6, None)
        
        theSession.SetUndoMarkName(markId5, "导入 STEP214 文件")
        
        step214Importer1.Destroy()
        
        markId7 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Start")
        
        step214Importer2 = theSession.DexManager.CreateStep214Importer()
        
        step214Importer2.SimplifyGeometry = True
        
        step214Importer2.LayerDefault = 1
        
        step214Importer2.OutputFile = "G:\\PHD2\\Code\\e2e_ablation\\scripts\\iou_3d\\iou_3d_strip_section_type_2.prt"
        
        step214Importer2.SettingsFile = "D:\\NX1953\\NX1953\\step214ug\\step214ug.def"
        
        step214Importer2.ObjectTypes.ProductData = True
        
        step214Importer2.OutputFile = ""
        
        theSession.SetUndoMarkName(markId7, "导入 STEP214 文件 对话框")
        
        # ----------------------------------------------
        #   对话开始 导入 STEP214 文件
        # ----------------------------------------------
        step214Importer2.Destroy()
        
        theSession.UndoToMark(markId7, None)
        
        theSession.DeleteUndoMark(markId7, None)
        
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
        
        curves1 = [NXOpen.IBaseCurve.Null] * 16 
        arc1 = workPart.Arcs.FindObject("ENTITY 5 6 1")
        curves1[0] = arc1
        arc2 = workPart.Arcs.FindObject("ENTITY 5 1 1")
        curves1[1] = arc2
        arc3 = workPart.Arcs.FindObject("ENTITY 5 10 1")
        curves1[2] = arc3
        arc4 = workPart.Arcs.FindObject("ENTITY 5 2 1")
        curves1[3] = arc4
        line1 = workPart.Lines.FindObject("ENTITY 3 2 1")
        curves1[4] = line1
        arc5 = workPart.Arcs.FindObject("ENTITY 5 7 1")
        curves1[5] = arc5
        arc6 = workPart.Arcs.FindObject("ENTITY 5 5 1")
        curves1[6] = arc6
        arc7 = workPart.Arcs.FindObject("ENTITY 5 11 1")
        curves1[7] = arc7
        line2 = workPart.Lines.FindObject("ENTITY 3 3 1")
        curves1[8] = line2
        arc8 = workPart.Arcs.FindObject("ENTITY 5 8 1")
        curves1[9] = arc8
        arc9 = workPart.Arcs.FindObject("ENTITY 5 3 1")
        curves1[10] = arc9
        arc10 = workPart.Arcs.FindObject("ENTITY 5 12 1")
        curves1[11] = arc10
        line3 = workPart.Lines.FindObject("ENTITY 3 4 1")
        curves1[12] = line3
        arc11 = workPart.Arcs.FindObject("ENTITY 5 9 1")
        curves1[13] = arc11
        line4 = workPart.Lines.FindObject("ENTITY 3 1 1")
        curves1[14] = line4
        arc12 = workPart.Arcs.FindObject("ENTITY 5 4 1")
        curves1[15] = arc12
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
        
        objects2 = [NXOpen.DisplayableObject.Null] * 16 
        compositeCurve1 = nXObject3
        arc13 = compositeCurve1.FindObject("CURVE 13 {5 (-0.3912862818355,-5.009291912854,0)}")
        objects2[0] = arc13
        arc14 = compositeCurve1.FindObject("CURVE 2 {5 (-2.7982458486376,9.578404262982,-0)}")
        objects2[1] = arc14
        arc15 = compositeCurve1.FindObject("CURVE 9 {5 (-2.5597674298829,1.0754194004682,0)}")
        objects2[2] = arc15
        arc16 = compositeCurve1.FindObject("CURVE 16 {5 (-2.7702464865025,-7.4551508200582,0)}")
        objects2[3] = arc16
        arc17 = compositeCurve1.FindObject("CURVE 5 {5 (-0.3303754855487,7.4990309628211,0)}")
        objects2[4] = arc17
        line5 = compositeCurve1.FindObject("CURVE 12 {3 (-1.3721345168131,-5.9356702788398,0)}")
        objects2[5] = line5
        arc18 = compositeCurve1.FindObject("CURVE 1 {5 (-3.7575594561337,1.0754194004682,0)}")
        objects2[6] = arc18
        arc19 = compositeCurve1.FindObject("CURVE 8 {5 (-1.9458607365815,8.7368905323186,0)}")
        objects2[7] = arc19
        arc20 = compositeCurve1.FindObject("CURVE 15 {5 (-1.0409002510708,-7.6352891479658,0)}")
        objects2[8] = arc20
        line6 = compositeCurve1.FindObject("CURVE 4 {3 (-0.2931128405779,8.7604942913567,0)}")
        objects2[9] = line6
        arc21 = compositeCurve1.FindObject("CURVE 11 {5 (-1.6735967925619,-6.6182341718042,0)}")
        objects2[10] = arc21
        arc22 = compositeCurve1.FindObject("CURVE 7 {5 (-1.6894926617046,8.7780670921438,0)}")
        objects2[11] = arc22
        line7 = compositeCurve1.FindObject("CURVE 14 {3 (-0.248613917961,-6.3508897414195,0)}")
        objects2[12] = line7
        arc23 = compositeCurve1.FindObject("CURVE 3 {5 (-1.1435011876657,9.8441810862489,0)}")
        objects2[13] = arc23
        arc24 = compositeCurve1.FindObject("CURVE 10 {5 (-1.9415228210483,-6.5903255056356,0)}")
        objects2[14] = arc24
        line8 = compositeCurve1.FindObject("CURVE 6 {3 (-1.3637634106659,8.2234532362034,0)}")
        objects2[15] = line8
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
        
        objects3 = [NXOpen.TaggedObject.Null] * 16 
        objects3[0] = line3
        objects3[1] = line2
        objects3[2] = line1
        objects3[3] = line4
        objects3[4] = arc10
        objects3[5] = arc7
        objects3[6] = arc3
        objects3[7] = arc11
        objects3[8] = arc8
        objects3[9] = arc5
        objects3[10] = arc1
        objects3[11] = arc6
        objects3[12] = arc12
        objects3[13] = arc9
        objects3[14] = arc4
        objects3[15] = arc2
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
        
        objects4 = [NXOpen.NXObject.Null] * 16 
        objects4[0] = arc18
        objects4[1] = arc14
        objects4[2] = arc23
        objects4[3] = line6
        objects4[4] = arc17
        objects4[5] = line8
        objects4[6] = arc22
        objects4[7] = arc19
        objects4[8] = arc15
        objects4[9] = arc24
        objects4[10] = arc21
        objects4[11] = line5
        objects4[12] = arc13
        objects4[13] = line7
        objects4[14] = arc20
        objects4[15] = arc16
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
        
        scaleAboutPoint1 = NXOpen.Point3d(-3.1872912024003268, -21.425679749469044, 0.0)
        viewCenter1 = NXOpen.Point3d(3.1872912024003268, 21.425679749469044, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint1, viewCenter1)
        
        scaleAboutPoint2 = NXOpen.Point3d(-1.9832034148268773, -16.857229026028541, 0.0)
        viewCenter2 = NXOpen.Point3d(1.9832034148268773, 16.857229026028541, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint2, viewCenter2)
        
        scaleAboutPoint3 = NXOpen.Point3d(0.90660727534946772, -13.485783220822807, 0.0)
        viewCenter3 = NXOpen.Point3d(-0.90660727534946772, 13.48578322082286, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint3, viewCenter3)
        
        scaleAboutPoint4 = NXOpen.Point3d(3.9890720115375085, -11.15126948679803, 0.0)
        viewCenter4 = NXOpen.Point3d(-3.9890720115375085, 11.151269486798052, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint4, viewCenter4)
        
        scaleAboutPoint5 = NXOpen.Point3d(6.9627438746836727, -8.6309012613265956, 0.0)
        viewCenter5 = NXOpen.Point3d(-6.9627438746836727, 8.6309012613266294, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint5, viewCenter5)
        
        scaleAboutPoint6 = NXOpen.Point3d(7.7750639933967571, -6.3244923528376527, 0.0)
        viewCenter6 = NXOpen.Point3d(-7.7750639933967571, 6.3244923528376527, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint6, viewCenter6)
        
        scaleAboutPoint7 = NXOpen.Point3d(7.1484170446752158, -4.4097377872996395, 0.0)
        viewCenter7 = NXOpen.Point3d(-7.1484170446752158, 4.4097377872996608, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint7, viewCenter7)
        
        scaleAboutPoint8 = NXOpen.Point3d(6.7585033876929437, -2.0424048699072044, 0.0)
        viewCenter8 = NXOpen.Point3d(-6.7585033876929437, 2.0424048699072044, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint8, viewCenter8)
        
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
        helpPoint2 = NXOpen.Point3d(3.9424896871135857, 0.021586363406375125, -0.0022035086303765113)
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
        objects8[0] = point46
        objects8[1] = point47
        objects8[2] = point48
        objects8[3] = point49
        objects8[4] = point50
        objects8[5] = point51
        objects8[6] = point52
        objects8[7] = point53
        objects8[8] = point54
        objects8[9] = point55
        objects8[10] = point56
        objects8[11] = spline1
        objects8[12] = point198
        objects8[13] = point199
        objects8[14] = point200
        objects8[15] = point201
        objects8[16] = point202
        objects8[17] = point203
        objects8[18] = point204
        objects8[19] = point205
        objects8[20] = point206
        objects8[21] = point207
        objects8[22] = point208
        objects8[23] = point209
        objects8[24] = point210
        objects8[25] = point211
        objects8[26] = point212
        objects8[27] = point213
        objects8[28] = point214
        objects8[29] = point215
        objects8[30] = point216
        objects8[31] = point217
        objects8[32] = point218
        objects8[33] = point219
        objects8[34] = point220
        objects8[35] = point221
        objects8[36] = point222
        objects8[37] = point223
        objects8[38] = point224
        objects8[39] = point93
        objects8[40] = point94
        objects8[41] = point95
        objects8[42] = point96
        objects8[43] = point97
        objects8[44] = point98
        objects8[45] = point99
        objects8[46] = point100
        objects8[47] = point101
        objects8[48] = point102
        objects8[49] = point103
        objects8[50] = point104
        objects8[51] = point105
        objects8[52] = point106
        objects8[53] = point107
        objects8[54] = point108
        objects8[55] = point109
        objects8[56] = point110
        objects8[57] = point111
        objects8[58] = point112
        objects8[59] = point113
        objects8[60] = point114
        objects8[61] = point115
        objects8[62] = point116
        objects8[63] = point117
        objects8[64] = point118
        objects8[65] = point119
        objects8[66] = point120
        objects8[67] = point121
        objects8[68] = point122
        objects8[69] = point123
        objects8[70] = point124
        objects8[71] = point125
        objects8[72] = point162
        objects8[73] = point163
        objects8[74] = point164
        objects8[75] = point165
        objects8[76] = point166
        objects8[77] = point167
        objects8[78] = point168
        objects8[79] = point169
        objects8[80] = point170
        objects8[81] = point171
        objects8[82] = point172
        objects8[83] = point173
        objects8[84] = point174
        objects8[85] = point175
        objects8[86] = point176
        objects8[87] = point177
        objects8[88] = point178
        objects8[89] = point179
        objects8[90] = point180
        objects8[91] = point181
        objects8[92] = point182
        objects8[93] = point183
        objects8[94] = point184
        objects8[95] = point185
        objects8[96] = point186
        objects8[97] = point187
        objects8[98] = point188
        objects8[99] = point189
        objects8[100] = point190
        objects8[101] = point191
        objects8[102] = point192
        objects8[103] = point193
        objects8[104] = point194
        objects8[105] = point195
        objects8[106] = point196
        objects8[107] = point197
        objects8[108] = point42
        objects8[109] = point43
        objects8[110] = point44
        objects8[111] = point45
        objects8[112] = point66
        objects8[113] = point67
        objects8[114] = point68
        objects8[115] = point69
        objects8[116] = point70
        objects8[117] = point71
        objects8[118] = point72
        objects8[119] = point73
        objects8[120] = point74
        objects8[121] = point75
        objects8[122] = point76
        objects8[123] = point77
        objects8[124] = point78
        objects8[125] = point79
        objects8[126] = point80
        objects8[127] = point81
        objects8[128] = point82
        objects8[129] = point83
        objects8[130] = point84
        objects8[131] = point85
        objects8[132] = point86
        objects8[133] = point87
        objects8[134] = point88
        objects8[135] = point89
        objects8[136] = point90
        objects8[137] = point91
        objects8[138] = point92
        objects8[139] = point286
        objects8[140] = point287
        objects8[141] = point288
        objects8[142] = point289
        objects8[143] = point290
        objects8[144] = point291
        objects8[145] = point292
        objects8[146] = point293
        objects8[147] = point294
        objects8[148] = point295
        objects8[149] = point296
        objects8[150] = point297
        objects8[151] = point298
        objects8[152] = point299
        objects8[153] = point300
        objects8[154] = point301
        objects8[155] = point302
        objects8[156] = point303
        objects8[157] = point304
        objects8[158] = point305
        objects8[159] = point306
        objects8[160] = group1
        objects8[161] = point57
        objects8[162] = point58
        objects8[163] = point59
        objects8[164] = point60
        objects8[165] = point61
        objects8[166] = point62
        objects8[167] = point63
        objects8[168] = point64
        objects8[169] = point65
        objects8[170] = point225
        objects8[171] = point226
        objects8[172] = point227
        objects8[173] = point228
        objects8[174] = point229
        objects8[175] = point230
        objects8[176] = point231
        objects8[177] = point232
        objects8[178] = point233
        objects8[179] = point234
        objects8[180] = point235
        objects8[181] = point236
        objects8[182] = point237
        objects8[183] = point238
        objects8[184] = point239
        objects8[185] = point240
        objects8[186] = point241
        objects8[187] = point242
        objects8[188] = point243
        objects8[189] = point244
        objects8[190] = point245
        objects8[191] = point246
        objects8[192] = point247
        objects8[193] = point248
        objects8[194] = point249
        objects8[195] = point250
        objects8[196] = point251
        objects8[197] = point252
        objects8[198] = point253
        objects8[199] = point254
        objects8[200] = point255
        objects8[201] = point256
        objects8[202] = point257
        objects8[203] = point258
        objects8[204] = point259
        objects8[205] = point260
        objects8[206] = point261
        objects8[207] = point262
        objects8[208] = point263
        objects8[209] = point264
        objects8[210] = point265
        objects8[211] = point266
        objects8[212] = point267
        objects8[213] = point268
        objects8[214] = point269
        objects8[215] = point270
        objects8[216] = point271
        objects8[217] = point272
        objects8[218] = point273
        objects8[219] = point274
        objects8[220] = point275
        objects8[221] = point276
        objects8[222] = point277
        objects8[223] = point278
        objects8[224] = point279
        objects8[225] = point280
        objects8[226] = point281
        objects8[227] = point282
        objects8[228] = point283
        objects8[229] = point284
        objects8[230] = point285
        objects8[231] = point6
        objects8[232] = point7
        objects8[233] = point8
        objects8[234] = point9
        objects8[235] = point10
        objects8[236] = point11
        objects8[237] = point12
        objects8[238] = point13
        objects8[239] = point14
        objects8[240] = point15
        objects8[241] = point16
        objects8[242] = point17
        objects8[243] = point18
        objects8[244] = point19
        objects8[245] = point20
        objects8[246] = point21
        objects8[247] = point22
        objects8[248] = point23
        objects8[249] = point24
        objects8[250] = point25
        objects8[251] = point26
        objects8[252] = point27
        objects8[253] = point28
        objects8[254] = point29
        objects8[255] = point30
        objects8[256] = point31
        objects8[257] = point32
        objects8[258] = point33
        objects8[259] = point34
        objects8[260] = point35
        objects8[261] = point36
        objects8[262] = point37
        objects8[263] = point38
        objects8[264] = point39
        objects8[265] = point40
        objects8[266] = point41
        objects8[267] = point126
        objects8[268] = point127
        objects8[269] = point128
        objects8[270] = point129
        objects8[271] = point130
        objects8[272] = point131
        objects8[273] = point132
        objects8[274] = point133
        objects8[275] = point134
        objects8[276] = point135
        objects8[277] = point136
        objects8[278] = point137
        objects8[279] = point138
        objects8[280] = point139
        objects8[281] = point140
        objects8[282] = point141
        objects8[283] = point142
        objects8[284] = point143
        objects8[285] = point144
        objects8[286] = point145
        objects8[287] = point146
        objects8[288] = point147
        objects8[289] = point148
        objects8[290] = point149
        objects8[291] = point150
        objects8[292] = point151
        objects8[293] = point152
        objects8[294] = point153
        objects8[295] = point154
        objects8[296] = point155
        objects8[297] = point156
        objects8[298] = point157
        objects8[299] = point158
        objects8[300] = point159
        objects8[301] = point160
        objects8[302] = point161
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
        face1 = swept1.FindObject("FACE 10015 {(262.2915677130831,2.6581970204865,-0.2262521988874) SWEPT(3)}")
        objects9[0] = face1
        displayModification1.Apply(objects9)
        
        face1.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects10 = [NXOpen.DisplayableObject.Null] * 1 
        face2 = swept1.FindObject("FACE 10014 {(262.3514985677187,1.7336082979685,0.7544507743374) SWEPT(3)}")
        objects10[0] = face2
        displayModification1.Apply(objects10)
        
        face2.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects11 = [NXOpen.DisplayableObject.Null] * 1 
        face3 = swept1.FindObject("FACE 10013 {(262.3936744800214,1.0523054818288,1.0558108604208) SWEPT(3)}")
        objects11[0] = face3
        displayModification1.Apply(objects11)
        
        face3.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects12 = [NXOpen.DisplayableObject.Null] * 1 
        face4 = swept1.FindObject("FACE 10012 {(262.3932684227858,1.080200872723,1.3237378468056) SWEPT(3)}")
        objects12[0] = face4
        displayModification1.Apply(objects12)
        
        face4.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects13 = [NXOpen.DisplayableObject.Null] * 1 
        face5 = swept1.FindObject("FACE 10011 {(261.9384349686234,8.7323298734859,1.943084118588) SWEPT(3)}")
        objects13[0] = face5
        displayModification1.Apply(objects13)
        
        face5.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects14 = [NXOpen.DisplayableObject.Null] * 1 
        face6 = swept1.FindObject("FACE 10010 {(261.4780601250592,16.3800244035669,1.330292628965) SWEPT(3)}")
        objects14[0] = face6
        displayModification1.Apply(objects14)
        
        face6.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects15 = [NXOpen.DisplayableObject.Null] * 1 
        face7 = swept1.FindObject("FACE 10009 {(261.4743952940463,16.4210923472313,1.0739334558078) SWEPT(3)}")
        objects15[0] = face7
        displayModification1.Apply(objects15)
        
        face7.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects16 = [NXOpen.DisplayableObject.Null] * 1 
        face8 = swept1.FindObject("FACE 10008 {(261.5059803824963,15.867425207377,0.7481277288743) SWEPT(3)}")
        objects16[0] = face8
        displayModification1.Apply(objects16)
        
        face8.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects17 = [NXOpen.DisplayableObject.Null] * 1 
        face9 = swept1.FindObject("FACE 10007 {(261.5443760405748,15.1441564557686,-0.2853531012528) SWEPT(3)}")
        objects17[0] = face9
        displayModification1.Apply(objects17)
        
        face9.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects18 = [NXOpen.DisplayableObject.Null] * 1 
        face10 = swept1.FindObject("FACE 10006 {(261.4688754207743,16.4033601816499,-0.3224328609676) SWEPT(3)}")
        objects18[0] = face10
        displayModification1.Apply(objects18)
        
        face10.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects19 = [NXOpen.DisplayableObject.Null] * 1 
        face11 = swept1.FindObject("FACE 10005 {(261.4081662142182,17.4852263690825,0.5281024577509) SWEPT(3)}")
        objects19[0] = face11
        displayModification1.Apply(objects19)
        
        face11.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects20 = [NXOpen.DisplayableObject.Null] * 1 
        face12 = swept1.FindObject("FACE 10004 {(261.431821107889,17.2201506159942,2.1827896619852) SWEPT(3)}")
        objects20[0] = face12
        displayModification1.Apply(objects20)
        
        face12.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects21 = [NXOpen.DisplayableObject.Null] * 1 
        face13 = swept1.FindObject("FACE 10003 {(261.9440699150994,8.7324935139761,3.1408623807818) SWEPT(3)}")
        objects21[0] = face13
        displayModification1.Apply(objects21)
        
        face13.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects22 = [NXOpen.DisplayableObject.Null] * 1 
        face14 = swept1.FindObject("FACE 10016 {(262.3710068820159,1.3189774300761,-0.3691169709414) SWEPT(3)}")
        objects22[0] = face14
        displayModification1.Apply(objects22)
        
        face14.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects23 = [NXOpen.DisplayableObject.Null] * 1 
        face15 = swept1.FindObject("FACE 2 {(522.7086093703734,34.0253243557089,-0.1390083222137) SWEPT(3)}")
        objects23[0] = face15
        displayModification1.Apply(objects23)
        
        face15.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects24 = [NXOpen.DisplayableObject.Null] * 1 
        face16 = swept1.FindObject("FACE 10002 {(262.448808119882,0.21703440993,2.1523269015845) SWEPT(3)}")
        objects24[0] = face16
        displayModification1.Apply(objects24)
        
        face16.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects25 = [NXOpen.DisplayableObject.Null] * 1 
        face17 = swept1.FindObject("FACE 1 {(0,1.0818839027918,1.9571319661241) SWEPT(3)}")
        objects25[0] = face17
        displayModification1.Apply(objects25)
        
        face17.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects26 = [NXOpen.DisplayableObject.Null] * 1 
        face18 = swept1.FindObject("FACE 10001 {(262.4514290431105,0.0369817681565,0.4229744833347) SWEPT(3)}")
        objects26[0] = face18
        displayModification1.Apply(objects26)
        
        face18.Color = 32767
        
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
        
        objects27 = [NXOpen.DisplayableObject.Null] * 1 
        body1 = workPart.Bodies.FindObject("SWEPT(3)")
        objects27[0] = body1
        theSession.DisplayManager.BlankObjects(objects27)
        
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
        objects28 = [NXOpen.TaggedObject.Null] * 302 
        point309 = workPart.Points.FindObject("ENTITY 2 301 1")
        objects28[0] = point309
        point310 = workPart.Points.FindObject("ENTITY 2 300 1")
        objects28[1] = point310
        point311 = workPart.Points.FindObject("ENTITY 2 299 1")
        objects28[2] = point311
        point312 = workPart.Points.FindObject("ENTITY 2 298 1")
        objects28[3] = point312
        point313 = workPart.Points.FindObject("ENTITY 2 297 1")
        objects28[4] = point313
        point314 = workPart.Points.FindObject("ENTITY 2 296 1")
        objects28[5] = point314
        point315 = workPart.Points.FindObject("ENTITY 2 295 1")
        objects28[6] = point315
        point316 = workPart.Points.FindObject("ENTITY 2 294 1")
        objects28[7] = point316
        point317 = workPart.Points.FindObject("ENTITY 2 293 1")
        objects28[8] = point317
        point318 = workPart.Points.FindObject("ENTITY 2 292 1")
        objects28[9] = point318
        point319 = workPart.Points.FindObject("ENTITY 2 291 1")
        objects28[10] = point319
        point320 = workPart.Points.FindObject("ENTITY 2 290 1")
        objects28[11] = point320
        point321 = workPart.Points.FindObject("ENTITY 2 289 1")
        objects28[12] = point321
        point322 = workPart.Points.FindObject("ENTITY 2 288 1")
        objects28[13] = point322
        point323 = workPart.Points.FindObject("ENTITY 2 287 1")
        objects28[14] = point323
        point324 = workPart.Points.FindObject("ENTITY 2 286 1")
        objects28[15] = point324
        point325 = workPart.Points.FindObject("ENTITY 2 285 1")
        objects28[16] = point325
        point326 = workPart.Points.FindObject("ENTITY 2 284 1")
        objects28[17] = point326
        point327 = workPart.Points.FindObject("ENTITY 2 283 1")
        objects28[18] = point327
        point328 = workPart.Points.FindObject("ENTITY 2 282 1")
        objects28[19] = point328
        point329 = workPart.Points.FindObject("ENTITY 2 281 1")
        objects28[20] = point329
        point330 = workPart.Points.FindObject("ENTITY 2 280 1")
        objects28[21] = point330
        point331 = workPart.Points.FindObject("ENTITY 2 279 1")
        objects28[22] = point331
        point332 = workPart.Points.FindObject("ENTITY 2 278 1")
        objects28[23] = point332
        point333 = workPart.Points.FindObject("ENTITY 2 277 1")
        objects28[24] = point333
        point334 = workPart.Points.FindObject("ENTITY 2 276 1")
        objects28[25] = point334
        point335 = workPart.Points.FindObject("ENTITY 2 275 1")
        objects28[26] = point335
        point336 = workPart.Points.FindObject("ENTITY 2 274 1")
        objects28[27] = point336
        point337 = workPart.Points.FindObject("ENTITY 2 273 1")
        objects28[28] = point337
        point338 = workPart.Points.FindObject("ENTITY 2 272 1")
        objects28[29] = point338
        point339 = workPart.Points.FindObject("ENTITY 2 271 1")
        objects28[30] = point339
        point340 = workPart.Points.FindObject("ENTITY 2 270 1")
        objects28[31] = point340
        point341 = workPart.Points.FindObject("ENTITY 2 269 1")
        objects28[32] = point341
        point342 = workPart.Points.FindObject("ENTITY 2 268 1")
        objects28[33] = point342
        point343 = workPart.Points.FindObject("ENTITY 2 267 1")
        objects28[34] = point343
        point344 = workPart.Points.FindObject("ENTITY 2 266 1")
        objects28[35] = point344
        point345 = workPart.Points.FindObject("ENTITY 2 265 1")
        objects28[36] = point345
        point346 = workPart.Points.FindObject("ENTITY 2 264 1")
        objects28[37] = point346
        point347 = workPart.Points.FindObject("ENTITY 2 263 1")
        objects28[38] = point347
        point348 = workPart.Points.FindObject("ENTITY 2 262 1")
        objects28[39] = point348
        point349 = workPart.Points.FindObject("ENTITY 2 261 1")
        objects28[40] = point349
        point350 = workPart.Points.FindObject("ENTITY 2 260 1")
        objects28[41] = point350
        point351 = workPart.Points.FindObject("ENTITY 2 259 1")
        objects28[42] = point351
        point352 = workPart.Points.FindObject("ENTITY 2 258 1")
        objects28[43] = point352
        point353 = workPart.Points.FindObject("ENTITY 2 257 1")
        objects28[44] = point353
        point354 = workPart.Points.FindObject("ENTITY 2 256 1")
        objects28[45] = point354
        point355 = workPart.Points.FindObject("ENTITY 2 255 1")
        objects28[46] = point355
        point356 = workPart.Points.FindObject("ENTITY 2 254 1")
        objects28[47] = point356
        point357 = workPart.Points.FindObject("ENTITY 2 253 1")
        objects28[48] = point357
        point358 = workPart.Points.FindObject("ENTITY 2 252 1")
        objects28[49] = point358
        point359 = workPart.Points.FindObject("ENTITY 2 251 1")
        objects28[50] = point359
        point360 = workPart.Points.FindObject("ENTITY 2 250 1")
        objects28[51] = point360
        point361 = workPart.Points.FindObject("ENTITY 2 249 1")
        objects28[52] = point361
        point362 = workPart.Points.FindObject("ENTITY 2 248 1")
        objects28[53] = point362
        point363 = workPart.Points.FindObject("ENTITY 2 247 1")
        objects28[54] = point363
        point364 = workPart.Points.FindObject("ENTITY 2 246 1")
        objects28[55] = point364
        point365 = workPart.Points.FindObject("ENTITY 2 245 1")
        objects28[56] = point365
        point366 = workPart.Points.FindObject("ENTITY 2 244 1")
        objects28[57] = point366
        point367 = workPart.Points.FindObject("ENTITY 2 243 1")
        objects28[58] = point367
        point368 = workPart.Points.FindObject("ENTITY 2 242 1")
        objects28[59] = point368
        point369 = workPart.Points.FindObject("ENTITY 2 241 1")
        objects28[60] = point369
        point370 = workPart.Points.FindObject("ENTITY 2 240 1")
        objects28[61] = point370
        point371 = workPart.Points.FindObject("ENTITY 2 239 1")
        objects28[62] = point371
        point372 = workPart.Points.FindObject("ENTITY 2 238 1")
        objects28[63] = point372
        point373 = workPart.Points.FindObject("ENTITY 2 237 1")
        objects28[64] = point373
        point374 = workPart.Points.FindObject("ENTITY 2 236 1")
        objects28[65] = point374
        point375 = workPart.Points.FindObject("ENTITY 2 235 1")
        objects28[66] = point375
        point376 = workPart.Points.FindObject("ENTITY 2 234 1")
        objects28[67] = point376
        point377 = workPart.Points.FindObject("ENTITY 2 233 1")
        objects28[68] = point377
        point378 = workPart.Points.FindObject("ENTITY 2 232 1")
        objects28[69] = point378
        point379 = workPart.Points.FindObject("ENTITY 2 231 1")
        objects28[70] = point379
        point380 = workPart.Points.FindObject("ENTITY 2 230 1")
        objects28[71] = point380
        point381 = workPart.Points.FindObject("ENTITY 2 229 1")
        objects28[72] = point381
        point382 = workPart.Points.FindObject("ENTITY 2 228 1")
        objects28[73] = point382
        point383 = workPart.Points.FindObject("ENTITY 2 227 1")
        objects28[74] = point383
        point384 = workPart.Points.FindObject("ENTITY 2 226 1")
        objects28[75] = point384
        point385 = workPart.Points.FindObject("ENTITY 2 225 1")
        objects28[76] = point385
        point386 = workPart.Points.FindObject("ENTITY 2 224 1")
        objects28[77] = point386
        point387 = workPart.Points.FindObject("ENTITY 2 223 1")
        objects28[78] = point387
        point388 = workPart.Points.FindObject("ENTITY 2 222 1")
        objects28[79] = point388
        point389 = workPart.Points.FindObject("ENTITY 2 221 1")
        objects28[80] = point389
        point390 = workPart.Points.FindObject("ENTITY 2 220 1")
        objects28[81] = point390
        point391 = workPart.Points.FindObject("ENTITY 2 219 1")
        objects28[82] = point391
        point392 = workPart.Points.FindObject("ENTITY 2 218 1")
        objects28[83] = point392
        point393 = workPart.Points.FindObject("ENTITY 2 217 1")
        objects28[84] = point393
        point394 = workPart.Points.FindObject("ENTITY 2 216 1")
        objects28[85] = point394
        point395 = workPart.Points.FindObject("ENTITY 2 215 1")
        objects28[86] = point395
        point396 = workPart.Points.FindObject("ENTITY 2 214 1")
        objects28[87] = point396
        point397 = workPart.Points.FindObject("ENTITY 2 213 1")
        objects28[88] = point397
        point398 = workPart.Points.FindObject("ENTITY 2 212 1")
        objects28[89] = point398
        point399 = workPart.Points.FindObject("ENTITY 2 211 1")
        objects28[90] = point399
        point400 = workPart.Points.FindObject("ENTITY 2 210 1")
        objects28[91] = point400
        point401 = workPart.Points.FindObject("ENTITY 2 209 1")
        objects28[92] = point401
        point402 = workPart.Points.FindObject("ENTITY 2 208 1")
        objects28[93] = point402
        point403 = workPart.Points.FindObject("ENTITY 2 207 1")
        objects28[94] = point403
        point404 = workPart.Points.FindObject("ENTITY 2 206 1")
        objects28[95] = point404
        point405 = workPart.Points.FindObject("ENTITY 2 205 1")
        objects28[96] = point405
        point406 = workPart.Points.FindObject("ENTITY 2 204 1")
        objects28[97] = point406
        point407 = workPart.Points.FindObject("ENTITY 2 203 1")
        objects28[98] = point407
        point408 = workPart.Points.FindObject("ENTITY 2 202 1")
        objects28[99] = point408
        point409 = workPart.Points.FindObject("ENTITY 2 201 1")
        objects28[100] = point409
        point410 = workPart.Points.FindObject("ENTITY 2 200 1")
        objects28[101] = point410
        point411 = workPart.Points.FindObject("ENTITY 2 199 1")
        objects28[102] = point411
        point412 = workPart.Points.FindObject("ENTITY 2 198 1")
        objects28[103] = point412
        point413 = workPart.Points.FindObject("ENTITY 2 197 1")
        objects28[104] = point413
        point414 = workPart.Points.FindObject("ENTITY 2 196 1")
        objects28[105] = point414
        point415 = workPart.Points.FindObject("ENTITY 2 195 1")
        objects28[106] = point415
        point416 = workPart.Points.FindObject("ENTITY 2 194 1")
        objects28[107] = point416
        point417 = workPart.Points.FindObject("ENTITY 2 193 1")
        objects28[108] = point417
        point418 = workPart.Points.FindObject("ENTITY 2 192 1")
        objects28[109] = point418
        point419 = workPart.Points.FindObject("ENTITY 2 191 1")
        objects28[110] = point419
        point420 = workPart.Points.FindObject("ENTITY 2 190 1")
        objects28[111] = point420
        point421 = workPart.Points.FindObject("ENTITY 2 189 1")
        objects28[112] = point421
        point422 = workPart.Points.FindObject("ENTITY 2 188 1")
        objects28[113] = point422
        point423 = workPart.Points.FindObject("ENTITY 2 187 1")
        objects28[114] = point423
        point424 = workPart.Points.FindObject("ENTITY 2 186 1")
        objects28[115] = point424
        point425 = workPart.Points.FindObject("ENTITY 2 185 1")
        objects28[116] = point425
        point426 = workPart.Points.FindObject("ENTITY 2 184 1")
        objects28[117] = point426
        point427 = workPart.Points.FindObject("ENTITY 2 183 1")
        objects28[118] = point427
        point428 = workPart.Points.FindObject("ENTITY 2 182 1")
        objects28[119] = point428
        point429 = workPart.Points.FindObject("ENTITY 2 181 1")
        objects28[120] = point429
        point430 = workPart.Points.FindObject("ENTITY 2 180 1")
        objects28[121] = point430
        point431 = workPart.Points.FindObject("ENTITY 2 179 1")
        objects28[122] = point431
        point432 = workPart.Points.FindObject("ENTITY 2 178 1")
        objects28[123] = point432
        point433 = workPart.Points.FindObject("ENTITY 2 177 1")
        objects28[124] = point433
        point434 = workPart.Points.FindObject("ENTITY 2 176 1")
        objects28[125] = point434
        point435 = workPart.Points.FindObject("ENTITY 2 175 1")
        objects28[126] = point435
        point436 = workPart.Points.FindObject("ENTITY 2 174 1")
        objects28[127] = point436
        point437 = workPart.Points.FindObject("ENTITY 2 173 1")
        objects28[128] = point437
        point438 = workPart.Points.FindObject("ENTITY 2 172 1")
        objects28[129] = point438
        point439 = workPart.Points.FindObject("ENTITY 2 171 1")
        objects28[130] = point439
        point440 = workPart.Points.FindObject("ENTITY 2 170 1")
        objects28[131] = point440
        point441 = workPart.Points.FindObject("ENTITY 2 169 1")
        objects28[132] = point441
        point442 = workPart.Points.FindObject("ENTITY 2 168 1")
        objects28[133] = point442
        point443 = workPart.Points.FindObject("ENTITY 2 167 1")
        objects28[134] = point443
        point444 = workPart.Points.FindObject("ENTITY 2 166 1")
        objects28[135] = point444
        point445 = workPart.Points.FindObject("ENTITY 2 165 1")
        objects28[136] = point445
        point446 = workPart.Points.FindObject("ENTITY 2 164 1")
        objects28[137] = point446
        point447 = workPart.Points.FindObject("ENTITY 2 163 1")
        objects28[138] = point447
        point448 = workPart.Points.FindObject("ENTITY 2 162 1")
        objects28[139] = point448
        point449 = workPart.Points.FindObject("ENTITY 2 161 1")
        objects28[140] = point449
        point450 = workPart.Points.FindObject("ENTITY 2 160 1")
        objects28[141] = point450
        point451 = workPart.Points.FindObject("ENTITY 2 159 1")
        objects28[142] = point451
        point452 = workPart.Points.FindObject("ENTITY 2 158 1")
        objects28[143] = point452
        point453 = workPart.Points.FindObject("ENTITY 2 157 1")
        objects28[144] = point453
        point454 = workPart.Points.FindObject("ENTITY 2 156 1")
        objects28[145] = point454
        point455 = workPart.Points.FindObject("ENTITY 2 155 1")
        objects28[146] = point455
        point456 = workPart.Points.FindObject("ENTITY 2 154 1")
        objects28[147] = point456
        point457 = workPart.Points.FindObject("ENTITY 2 153 1")
        objects28[148] = point457
        point458 = workPart.Points.FindObject("ENTITY 2 152 1")
        objects28[149] = point458
        point459 = workPart.Points.FindObject("ENTITY 2 151 1")
        objects28[150] = point459
        point460 = workPart.Points.FindObject("ENTITY 2 150 1")
        objects28[151] = point460
        point461 = workPart.Points.FindObject("ENTITY 2 149 1")
        objects28[152] = point461
        point462 = workPart.Points.FindObject("ENTITY 2 148 1")
        objects28[153] = point462
        point463 = workPart.Points.FindObject("ENTITY 2 147 1")
        objects28[154] = point463
        point464 = workPart.Points.FindObject("ENTITY 2 146 1")
        objects28[155] = point464
        point465 = workPart.Points.FindObject("ENTITY 2 145 1")
        objects28[156] = point465
        point466 = workPart.Points.FindObject("ENTITY 2 144 1")
        objects28[157] = point466
        point467 = workPart.Points.FindObject("ENTITY 2 143 1")
        objects28[158] = point467
        point468 = workPart.Points.FindObject("ENTITY 2 142 1")
        objects28[159] = point468
        point469 = workPart.Points.FindObject("ENTITY 2 141 1")
        objects28[160] = point469
        point470 = workPart.Points.FindObject("ENTITY 2 140 1")
        objects28[161] = point470
        point471 = workPart.Points.FindObject("ENTITY 2 139 1")
        objects28[162] = point471
        point472 = workPart.Points.FindObject("ENTITY 2 138 1")
        objects28[163] = point472
        point473 = workPart.Points.FindObject("ENTITY 2 137 1")
        objects28[164] = point473
        point474 = workPart.Points.FindObject("ENTITY 2 136 1")
        objects28[165] = point474
        point475 = workPart.Points.FindObject("ENTITY 2 135 1")
        objects28[166] = point475
        point476 = workPart.Points.FindObject("ENTITY 2 134 1")
        objects28[167] = point476
        point477 = workPart.Points.FindObject("ENTITY 2 133 1")
        objects28[168] = point477
        point478 = workPart.Points.FindObject("ENTITY 2 132 1")
        objects28[169] = point478
        point479 = workPart.Points.FindObject("ENTITY 2 131 1")
        objects28[170] = point479
        point480 = workPart.Points.FindObject("ENTITY 2 130 1")
        objects28[171] = point480
        point481 = workPart.Points.FindObject("ENTITY 2 129 1")
        objects28[172] = point481
        point482 = workPart.Points.FindObject("ENTITY 2 128 1")
        objects28[173] = point482
        point483 = workPart.Points.FindObject("ENTITY 2 127 1")
        objects28[174] = point483
        point484 = workPart.Points.FindObject("ENTITY 2 126 1")
        objects28[175] = point484
        point485 = workPart.Points.FindObject("ENTITY 2 125 1")
        objects28[176] = point485
        point486 = workPart.Points.FindObject("ENTITY 2 124 1")
        objects28[177] = point486
        point487 = workPart.Points.FindObject("ENTITY 2 123 1")
        objects28[178] = point487
        point488 = workPart.Points.FindObject("ENTITY 2 122 1")
        objects28[179] = point488
        point489 = workPart.Points.FindObject("ENTITY 2 121 1")
        objects28[180] = point489
        point490 = workPart.Points.FindObject("ENTITY 2 120 1")
        objects28[181] = point490
        point491 = workPart.Points.FindObject("ENTITY 2 119 1")
        objects28[182] = point491
        point492 = workPart.Points.FindObject("ENTITY 2 118 1")
        objects28[183] = point492
        point493 = workPart.Points.FindObject("ENTITY 2 117 1")
        objects28[184] = point493
        point494 = workPart.Points.FindObject("ENTITY 2 116 1")
        objects28[185] = point494
        point495 = workPart.Points.FindObject("ENTITY 2 115 1")
        objects28[186] = point495
        point496 = workPart.Points.FindObject("ENTITY 2 114 1")
        objects28[187] = point496
        point497 = workPart.Points.FindObject("ENTITY 2 113 1")
        objects28[188] = point497
        point498 = workPart.Points.FindObject("ENTITY 2 112 1")
        objects28[189] = point498
        point499 = workPart.Points.FindObject("ENTITY 2 111 1")
        objects28[190] = point499
        point500 = workPart.Points.FindObject("ENTITY 2 110 1")
        objects28[191] = point500
        point501 = workPart.Points.FindObject("ENTITY 2 109 1")
        objects28[192] = point501
        point502 = workPart.Points.FindObject("ENTITY 2 108 1")
        objects28[193] = point502
        point503 = workPart.Points.FindObject("ENTITY 2 107 1")
        objects28[194] = point503
        point504 = workPart.Points.FindObject("ENTITY 2 106 1")
        objects28[195] = point504
        point505 = workPart.Points.FindObject("ENTITY 2 105 1")
        objects28[196] = point505
        point506 = workPart.Points.FindObject("ENTITY 2 104 1")
        objects28[197] = point506
        point507 = workPart.Points.FindObject("ENTITY 2 103 1")
        objects28[198] = point507
        point508 = workPart.Points.FindObject("ENTITY 2 102 1")
        objects28[199] = point508
        point509 = workPart.Points.FindObject("ENTITY 2 101 1")
        objects28[200] = point509
        point510 = workPart.Points.FindObject("ENTITY 2 100 1")
        objects28[201] = point510
        point511 = workPart.Points.FindObject("ENTITY 2 99 1")
        objects28[202] = point511
        point512 = workPart.Points.FindObject("ENTITY 2 98 1")
        objects28[203] = point512
        point513 = workPart.Points.FindObject("ENTITY 2 97 1")
        objects28[204] = point513
        point514 = workPart.Points.FindObject("ENTITY 2 96 1")
        objects28[205] = point514
        point515 = workPart.Points.FindObject("ENTITY 2 95 1")
        objects28[206] = point515
        point516 = workPart.Points.FindObject("ENTITY 2 94 1")
        objects28[207] = point516
        point517 = workPart.Points.FindObject("ENTITY 2 93 1")
        objects28[208] = point517
        point518 = workPart.Points.FindObject("ENTITY 2 92 1")
        objects28[209] = point518
        point519 = workPart.Points.FindObject("ENTITY 2 91 1")
        objects28[210] = point519
        point520 = workPart.Points.FindObject("ENTITY 2 90 1")
        objects28[211] = point520
        point521 = workPart.Points.FindObject("ENTITY 2 89 1")
        objects28[212] = point521
        point522 = workPart.Points.FindObject("ENTITY 2 88 1")
        objects28[213] = point522
        point523 = workPart.Points.FindObject("ENTITY 2 87 1")
        objects28[214] = point523
        point524 = workPart.Points.FindObject("ENTITY 2 86 1")
        objects28[215] = point524
        point525 = workPart.Points.FindObject("ENTITY 2 85 1")
        objects28[216] = point525
        point526 = workPart.Points.FindObject("ENTITY 2 84 1")
        objects28[217] = point526
        point527 = workPart.Points.FindObject("ENTITY 2 83 1")
        objects28[218] = point527
        point528 = workPart.Points.FindObject("ENTITY 2 82 1")
        objects28[219] = point528
        point529 = workPart.Points.FindObject("ENTITY 2 81 1")
        objects28[220] = point529
        point530 = workPart.Points.FindObject("ENTITY 2 80 1")
        objects28[221] = point530
        point531 = workPart.Points.FindObject("ENTITY 2 79 1")
        objects28[222] = point531
        point532 = workPart.Points.FindObject("ENTITY 2 78 1")
        objects28[223] = point532
        point533 = workPart.Points.FindObject("ENTITY 2 77 1")
        objects28[224] = point533
        point534 = workPart.Points.FindObject("ENTITY 2 76 1")
        objects28[225] = point534
        point535 = workPart.Points.FindObject("ENTITY 2 75 1")
        objects28[226] = point535
        point536 = workPart.Points.FindObject("ENTITY 2 74 1")
        objects28[227] = point536
        point537 = workPart.Points.FindObject("ENTITY 2 73 1")
        objects28[228] = point537
        point538 = workPart.Points.FindObject("ENTITY 2 72 1")
        objects28[229] = point538
        point539 = workPart.Points.FindObject("ENTITY 2 71 1")
        objects28[230] = point539
        point540 = workPart.Points.FindObject("ENTITY 2 70 1")
        objects28[231] = point540
        point541 = workPart.Points.FindObject("ENTITY 2 69 1")
        objects28[232] = point541
        point542 = workPart.Points.FindObject("ENTITY 2 68 1")
        objects28[233] = point542
        point543 = workPart.Points.FindObject("ENTITY 2 67 1")
        objects28[234] = point543
        point544 = workPart.Points.FindObject("ENTITY 2 66 1")
        objects28[235] = point544
        point545 = workPart.Points.FindObject("ENTITY 2 65 1")
        objects28[236] = point545
        point546 = workPart.Points.FindObject("ENTITY 2 64 1")
        objects28[237] = point546
        point547 = workPart.Points.FindObject("ENTITY 2 63 1")
        objects28[238] = point547
        point548 = workPart.Points.FindObject("ENTITY 2 62 1")
        objects28[239] = point548
        point549 = workPart.Points.FindObject("ENTITY 2 61 1")
        objects28[240] = point549
        point550 = workPart.Points.FindObject("ENTITY 2 60 1")
        objects28[241] = point550
        point551 = workPart.Points.FindObject("ENTITY 2 59 1")
        objects28[242] = point551
        point552 = workPart.Points.FindObject("ENTITY 2 58 1")
        objects28[243] = point552
        point553 = workPart.Points.FindObject("ENTITY 2 57 1")
        objects28[244] = point553
        point554 = workPart.Points.FindObject("ENTITY 2 56 1")
        objects28[245] = point554
        point555 = workPart.Points.FindObject("ENTITY 2 55 1")
        objects28[246] = point555
        point556 = workPart.Points.FindObject("ENTITY 2 54 1")
        objects28[247] = point556
        point557 = workPart.Points.FindObject("ENTITY 2 53 1")
        objects28[248] = point557
        point558 = workPart.Points.FindObject("ENTITY 2 52 1")
        objects28[249] = point558
        point559 = workPart.Points.FindObject("ENTITY 2 51 1")
        objects28[250] = point559
        point560 = workPart.Points.FindObject("ENTITY 2 50 1")
        objects28[251] = point560
        point561 = workPart.Points.FindObject("ENTITY 2 49 1")
        objects28[252] = point561
        point562 = workPart.Points.FindObject("ENTITY 2 48 1")
        objects28[253] = point562
        point563 = workPart.Points.FindObject("ENTITY 2 47 1")
        objects28[254] = point563
        point564 = workPart.Points.FindObject("ENTITY 2 46 1")
        objects28[255] = point564
        point565 = workPart.Points.FindObject("ENTITY 2 45 1")
        objects28[256] = point565
        point566 = workPart.Points.FindObject("ENTITY 2 44 1")
        objects28[257] = point566
        point567 = workPart.Points.FindObject("ENTITY 2 43 1")
        objects28[258] = point567
        point568 = workPart.Points.FindObject("ENTITY 2 42 1")
        objects28[259] = point568
        point569 = workPart.Points.FindObject("ENTITY 2 41 1")
        objects28[260] = point569
        point570 = workPart.Points.FindObject("ENTITY 2 40 1")
        objects28[261] = point570
        point571 = workPart.Points.FindObject("ENTITY 2 39 1")
        objects28[262] = point571
        point572 = workPart.Points.FindObject("ENTITY 2 38 1")
        objects28[263] = point572
        point573 = workPart.Points.FindObject("ENTITY 2 37 1")
        objects28[264] = point573
        point574 = workPart.Points.FindObject("ENTITY 2 36 1")
        objects28[265] = point574
        point575 = workPart.Points.FindObject("ENTITY 2 35 1")
        objects28[266] = point575
        point576 = workPart.Points.FindObject("ENTITY 2 34 1")
        objects28[267] = point576
        point577 = workPart.Points.FindObject("ENTITY 2 33 1")
        objects28[268] = point577
        point578 = workPart.Points.FindObject("ENTITY 2 32 1")
        objects28[269] = point578
        point579 = workPart.Points.FindObject("ENTITY 2 31 1")
        objects28[270] = point579
        point580 = workPart.Points.FindObject("ENTITY 2 30 1")
        objects28[271] = point580
        point581 = workPart.Points.FindObject("ENTITY 2 29 1")
        objects28[272] = point581
        point582 = workPart.Points.FindObject("ENTITY 2 28 1")
        objects28[273] = point582
        point583 = workPart.Points.FindObject("ENTITY 2 27 1")
        objects28[274] = point583
        point584 = workPart.Points.FindObject("ENTITY 2 26 1")
        objects28[275] = point584
        point585 = workPart.Points.FindObject("ENTITY 2 25 1")
        objects28[276] = point585
        point586 = workPart.Points.FindObject("ENTITY 2 24 1")
        objects28[277] = point586
        point587 = workPart.Points.FindObject("ENTITY 2 23 1")
        objects28[278] = point587
        point588 = workPart.Points.FindObject("ENTITY 2 22 1")
        objects28[279] = point588
        point589 = workPart.Points.FindObject("ENTITY 2 21 1")
        objects28[280] = point589
        point590 = workPart.Points.FindObject("ENTITY 2 20 1")
        objects28[281] = point590
        point591 = workPart.Points.FindObject("ENTITY 2 19 1")
        objects28[282] = point591
        point592 = workPart.Points.FindObject("ENTITY 2 18 1")
        objects28[283] = point592
        point593 = workPart.Points.FindObject("ENTITY 2 17 1")
        objects28[284] = point593
        point594 = workPart.Points.FindObject("ENTITY 2 16 1")
        objects28[285] = point594
        point595 = workPart.Points.FindObject("ENTITY 2 15 1")
        objects28[286] = point595
        point596 = workPart.Points.FindObject("ENTITY 2 14 1")
        objects28[287] = point596
        point597 = workPart.Points.FindObject("ENTITY 2 13 1")
        objects28[288] = point597
        point598 = workPart.Points.FindObject("ENTITY 2 12 1")
        objects28[289] = point598
        point599 = workPart.Points.FindObject("ENTITY 2 11 1")
        objects28[290] = point599
        point600 = workPart.Points.FindObject("ENTITY 2 10 1")
        objects28[291] = point600
        point601 = workPart.Points.FindObject("ENTITY 2 9 1")
        objects28[292] = point601
        point602 = workPart.Points.FindObject("ENTITY 2 8 1")
        objects28[293] = point602
        point603 = workPart.Points.FindObject("ENTITY 2 7 1")
        objects28[294] = point603
        point604 = workPart.Points.FindObject("ENTITY 2 6 1")
        objects28[295] = point604
        point605 = workPart.Points.FindObject("ENTITY 2 5 1")
        objects28[296] = point605
        point606 = workPart.Points.FindObject("ENTITY 2 4 1")
        objects28[297] = point606
        point607 = workPart.Points.FindObject("ENTITY 2 3 1")
        objects28[298] = point607
        point608 = workPart.Points.FindObject("ENTITY 2 2 1")
        objects28[299] = point608
        point609 = workPart.Points.FindObject("ENTITY 2 1 1")
        objects28[300] = point609
        group2 = nXObject9
        objects28[301] = group2
        added3 = fitCurveBuilder2.Target.Add(objects28)
        
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
        
        points1 = [NXOpen.Point.Null] * 1 
        points1[0] = point310
        curveDumbRule3 = workPart.ScRuleFactory.CreateRuleCurveDumbFromPoints(points1, selectionIntentRuleOptions5)
        
        selectionIntentRuleOptions5.Dispose()
        compositeCurveBuilder3.Section.AllowSelfIntersection(False)
        
        compositeCurveBuilder3.Section.AllowDegenerateCurves(False)
        
        rules5 = [None] * 1 
        rules5[0] = curveDumbRule3
        helpPoint5 = NXOpen.Point3d(1.7392414808273315, 0.00093553331680595875, 0.00068816542625427246)
        compositeCurveBuilder3.Section.AddToSection(rules5, point310, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint5, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId65, None)
        
        expression32 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        theSession.DeleteUndoMark(markId64, None)
        
        compositeCurveBuilder3.Destroy()
        
        waveDatumBuilder3.Destroy()
        
        wavePointBuilder3.Destroy()
        
        extractFaceBuilder3.Destroy()
        
        mirrorBodyBuilder3.Destroy()
        
        waveSketchBuilder3.Destroy()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression32)
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        theSession.UndoToMark(markId63, None)
        
        theSession.DeleteUndoMark(markId63, None)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->1 抽取几何特征
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->关联复制(A)->抽取几何特征(E)...
        # ----------------------------------------------
        markId66 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        wavePointBuilder4 = workPart.Features.CreateWavePointBuilder(NXOpen.Features.Feature.Null)
        
        waveDatumBuilder4 = workPart.Features.CreateWaveDatumBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder4 = workPart.Features.CreateCompositeCurveBuilder(NXOpen.Features.Feature.Null)
        
        extractFaceBuilder4 = workPart.Features.CreateExtractFaceBuilder(NXOpen.Features.Feature.Null)
        
        mirrorBodyBuilder4 = workPart.Features.CreateMirrorBodyBuilder(NXOpen.Features.Feature.Null)
        
        waveSketchBuilder4 = workPart.Features.CreateWaveSketchBuilder(NXOpen.Features.Feature.Null)
        
        compositeCurveBuilder4.CurveFitData.Tolerance = 0.001
        
        compositeCurveBuilder4.CurveFitData.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder4.Section.SetAllowRefCrvs(False)
        
        extractFaceBuilder4.FaceOption = NXOpen.Features.ExtractFaceBuilder.FaceOptionType.AdjacentFaces
        
        compositeCurveBuilder4.Associative = False
        
        waveDatumBuilder4.ParentPart = NXOpen.Features.WaveDatumBuilder.ParentPartType.WorkPart
        
        wavePointBuilder4.ParentPart = NXOpen.Features.WavePointBuilder.ParentPartType.WorkPart
        
        extractFaceBuilder4.ParentPart = NXOpen.Features.ExtractFaceBuilder.ParentPartType.WorkPart
        
        mirrorBodyBuilder4.ParentPartType = NXOpen.Features.MirrorBodyBuilder.ParentPart.WorkPart
        
        compositeCurveBuilder4.ParentPart = NXOpen.Features.CompositeCurveBuilder.PartType.WorkPart
        
        waveSketchBuilder4.ParentPart = NXOpen.Features.WaveSketchBuilder.ParentPartType.WorkPart
        
        compositeCurveBuilder4.Associative = False
        
        theSession.SetUndoMarkName(markId66, "抽取几何特征 对话框")
        
        compositeCurveBuilder4.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder4.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder4.Section.AngleTolerance = 0.050000000000000003
        
        compositeCurveBuilder4.Section.DistanceTolerance = 0.001
        
        compositeCurveBuilder4.Section.ChainingTolerance = 0.00095
        
        compositeCurveBuilder4.Associative = False
        
        compositeCurveBuilder4.FixAtCurrentTimestamp = False
        
        compositeCurveBuilder4.HideOriginal = False
        
        compositeCurveBuilder4.InheritDisplayProperties = False
        
        compositeCurveBuilder4.Section.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.CurvesAndPoints)
        
        markId67 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId68 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions6 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions6.SetSelectedFromInactive(False)
        
        curves3 = [NXOpen.IBaseCurve.Null] * 1 
        spline3 = workPart.Splines.FindObject("ENTITY 9 1 1")
        curves3[0] = spline3
        curveDumbRule4 = workPart.ScRuleFactory.CreateRuleBaseCurveDumb(curves3, selectionIntentRuleOptions6)
        
        selectionIntentRuleOptions6.Dispose()
        compositeCurveBuilder4.Section.AllowSelfIntersection(False)
        
        compositeCurveBuilder4.Section.AllowDegenerateCurves(False)
        
        rules6 = [None] * 1 
        rules6[0] = curveDumbRule4
        helpPoint6 = NXOpen.Point3d(2.7477171380833205, 0.0084519045758358821, 0.0010717850333356116)
        compositeCurveBuilder4.Section.AddToSection(rules6, spline3, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint6, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId68, None)
        
        theSession.DeleteUndoMark(markId67, None)
        
        markId69 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        theSession.DeleteUndoMark(markId69, None)
        
        markId70 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "抽取几何特征")
        
        nXObject11 = compositeCurveBuilder4.Commit()
        
        theSession.DeleteUndoMark(markId70, None)
        
        theSession.SetUndoMarkName(markId66, "抽取几何特征")
        
        compositeCurveBuilder4.Destroy()
        
        waveDatumBuilder4.Destroy()
        
        wavePointBuilder4.Destroy()
        
        extractFaceBuilder4.Destroy()
        
        mirrorBodyBuilder4.Destroy()
        
        waveSketchBuilder4.Destroy()
        
        markId71 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects29 = [NXOpen.DisplayableObject.Null] * 1 
        compositeCurve3 = nXObject11
        spline4 = compositeCurve3.FindObject("CURVE 1")
        objects29[0] = spline4
        theSession.DisplayManager.BlankObjects(objects29)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        # ----------------------------------------------
        #   菜单：编辑(E)->选择(L)->全选(S)
        # ----------------------------------------------
        # Refer to the sample NXOpen application, Selection for "Select All" alternatives.
        # ----------------------------------------------
        #   菜单：编辑(E)->删除(D)...
        # ----------------------------------------------
        markId72 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Delete")
        
        theSession.UpdateManager.ClearErrorList()
        
        markId73 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Delete")
        
        objects30 = [NXOpen.TaggedObject.Null] * 303 
        objects30[0] = point342
        objects30[1] = point343
        objects30[2] = point344
        objects30[3] = point345
        objects30[4] = point346
        objects30[5] = point347
        objects30[6] = point348
        objects30[7] = point349
        objects30[8] = point350
        objects30[9] = point351
        objects30[10] = point352
        objects30[11] = point495
        objects30[12] = point496
        objects30[13] = point497
        objects30[14] = point498
        objects30[15] = point499
        objects30[16] = point500
        objects30[17] = point501
        objects30[18] = point502
        objects30[19] = point503
        objects30[20] = point504
        objects30[21] = point505
        objects30[22] = point506
        objects30[23] = point507
        objects30[24] = point508
        objects30[25] = point509
        objects30[26] = point388
        objects30[27] = point389
        objects30[28] = point390
        objects30[29] = point391
        objects30[30] = point392
        objects30[31] = point393
        objects30[32] = point394
        objects30[33] = point395
        objects30[34] = point396
        objects30[35] = point397
        objects30[36] = point398
        objects30[37] = point399
        objects30[38] = point400
        objects30[39] = point401
        objects30[40] = point402
        objects30[41] = point403
        objects30[42] = point404
        objects30[43] = point405
        objects30[44] = point406
        objects30[45] = point407
        objects30[46] = point408
        objects30[47] = point409
        objects30[48] = point410
        objects30[49] = point411
        objects30[50] = point412
        objects30[51] = point413
        objects30[52] = point414
        objects30[53] = point415
        objects30[54] = point416
        objects30[55] = point417
        objects30[56] = point418
        objects30[57] = point419
        objects30[58] = point420
        objects30[59] = point421
        objects30[60] = point459
        objects30[61] = point460
        objects30[62] = point461
        objects30[63] = point462
        objects30[64] = point463
        objects30[65] = point464
        objects30[66] = point465
        objects30[67] = point466
        objects30[68] = point467
        objects30[69] = point468
        objects30[70] = point469
        objects30[71] = point470
        objects30[72] = point471
        objects30[73] = point472
        objects30[74] = point473
        objects30[75] = point474
        objects30[76] = point475
        objects30[77] = point476
        objects30[78] = point477
        objects30[79] = point478
        objects30[80] = point479
        objects30[81] = point480
        objects30[82] = point481
        objects30[83] = point482
        objects30[84] = point483
        objects30[85] = point484
        objects30[86] = point485
        objects30[87] = point486
        objects30[88] = point487
        objects30[89] = point488
        objects30[90] = point489
        objects30[91] = point490
        objects30[92] = point491
        objects30[93] = point492
        objects30[94] = point493
        objects30[95] = point494
        objects30[96] = point338
        objects30[97] = point339
        objects30[98] = point340
        objects30[99] = point341
        objects30[100] = point599
        objects30[101] = point600
        objects30[102] = point601
        objects30[103] = point602
        objects30[104] = point603
        objects30[105] = point604
        objects30[106] = point605
        objects30[107] = point606
        objects30[108] = point607
        objects30[109] = point608
        objects30[110] = point609
        objects30[111] = group2
        objects30[112] = spline3
        objects30[113] = point362
        objects30[114] = point363
        objects30[115] = point364
        objects30[116] = point365
        objects30[117] = point366
        objects30[118] = point367
        objects30[119] = point368
        objects30[120] = point369
        objects30[121] = point370
        objects30[122] = point371
        objects30[123] = point372
        objects30[124] = point373
        objects30[125] = point374
        objects30[126] = point375
        objects30[127] = point376
        objects30[128] = point377
        objects30[129] = point378
        objects30[130] = point379
        objects30[131] = point380
        objects30[132] = point381
        objects30[133] = point382
        objects30[134] = point383
        objects30[135] = point384
        objects30[136] = point385
        objects30[137] = point386
        objects30[138] = point387
        objects30[139] = point555
        objects30[140] = point556
        objects30[141] = point557
        objects30[142] = point558
        objects30[143] = point559
        objects30[144] = point560
        objects30[145] = point561
        objects30[146] = point562
        objects30[147] = point563
        objects30[148] = point564
        objects30[149] = point565
        objects30[150] = point566
        objects30[151] = point567
        objects30[152] = point568
        objects30[153] = point569
        objects30[154] = point570
        objects30[155] = point571
        objects30[156] = point572
        objects30[157] = point573
        objects30[158] = point574
        objects30[159] = point575
        objects30[160] = point576
        objects30[161] = point577
        objects30[162] = point578
        objects30[163] = point579
        objects30[164] = point580
        objects30[165] = point581
        objects30[166] = point582
        objects30[167] = point583
        objects30[168] = point584
        objects30[169] = point585
        objects30[170] = point586
        objects30[171] = point587
        objects30[172] = point588
        objects30[173] = point589
        objects30[174] = point590
        objects30[175] = point591
        objects30[176] = point592
        objects30[177] = point593
        objects30[178] = point594
        objects30[179] = point595
        objects30[180] = point596
        objects30[181] = point597
        objects30[182] = point598
        objects30[183] = point353
        objects30[184] = point354
        objects30[185] = point355
        objects30[186] = point356
        objects30[187] = point357
        objects30[188] = point358
        objects30[189] = point359
        objects30[190] = point360
        objects30[191] = point361
        objects30[192] = point309
        objects30[193] = point310
        objects30[194] = point311
        objects30[195] = point312
        objects30[196] = point313
        objects30[197] = point314
        objects30[198] = point315
        objects30[199] = point316
        objects30[200] = point317
        objects30[201] = point318
        objects30[202] = point319
        objects30[203] = point320
        objects30[204] = point321
        objects30[205] = point322
        objects30[206] = point323
        objects30[207] = point324
        objects30[208] = point325
        objects30[209] = point326
        objects30[210] = point327
        objects30[211] = point328
        objects30[212] = point329
        objects30[213] = point330
        objects30[214] = point331
        objects30[215] = point332
        objects30[216] = point333
        objects30[217] = point334
        objects30[218] = point335
        objects30[219] = point336
        objects30[220] = point337
        objects30[221] = point510
        objects30[222] = point511
        objects30[223] = point512
        objects30[224] = point513
        objects30[225] = point514
        objects30[226] = point515
        objects30[227] = point516
        objects30[228] = point517
        objects30[229] = point518
        objects30[230] = point519
        objects30[231] = point520
        objects30[232] = point521
        objects30[233] = point522
        objects30[234] = point523
        objects30[235] = point524
        objects30[236] = point525
        objects30[237] = point526
        objects30[238] = point527
        objects30[239] = point528
        objects30[240] = point529
        objects30[241] = point530
        objects30[242] = point531
        objects30[243] = point532
        objects30[244] = point533
        objects30[245] = point534
        objects30[246] = point535
        objects30[247] = point536
        objects30[248] = point537
        objects30[249] = point538
        objects30[250] = point539
        objects30[251] = point540
        objects30[252] = point541
        objects30[253] = point542
        objects30[254] = point543
        objects30[255] = point544
        objects30[256] = point545
        objects30[257] = point546
        objects30[258] = point547
        objects30[259] = point548
        objects30[260] = point549
        objects30[261] = point550
        objects30[262] = point551
        objects30[263] = point552
        objects30[264] = point553
        objects30[265] = point554
        objects30[266] = point422
        objects30[267] = point423
        objects30[268] = point424
        objects30[269] = point425
        objects30[270] = point426
        objects30[271] = point427
        objects30[272] = point428
        objects30[273] = point429
        objects30[274] = point430
        objects30[275] = point431
        objects30[276] = point432
        objects30[277] = point433
        objects30[278] = point434
        objects30[279] = point435
        objects30[280] = point436
        objects30[281] = point437
        objects30[282] = point438
        objects30[283] = point439
        objects30[284] = point440
        objects30[285] = point441
        objects30[286] = point442
        objects30[287] = point443
        objects30[288] = point444
        objects30[289] = point445
        objects30[290] = point446
        objects30[291] = point447
        objects30[292] = point448
        objects30[293] = point449
        objects30[294] = point450
        objects30[295] = point451
        objects30[296] = point452
        objects30[297] = point453
        objects30[298] = point454
        objects30[299] = point455
        objects30[300] = point456
        objects30[301] = point457
        objects30[302] = point458
        nErrs8 = theSession.UpdateManager.AddObjectsToDeleteList(objects30)
        
        id3 = theSession.NewestVisibleUndoMark
        
        nErrs9 = theSession.UpdateManager.DoUpdate(id3)
        
        theSession.DeleteUndoMark(markId72, None)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->5 扫掠
        # ----------------------------------------------
        # ----------------------------------------------
        #   菜单：插入(S)->扫掠(W)->扫掠(S)...
        # ----------------------------------------------
        markId74 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
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
        
        theSession.SetUndoMarkName(markId74, "扫掠 对话框")
        
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
        
        markId75 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId76 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions7 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions7.SetSelectedFromInactive(False)
        
        features3 = [NXOpen.Features.Feature.Null] * 1 
        features3[0] = compositeCurve3
        curveFeatureRule3 = workPart.ScRuleFactory.CreateRuleCurveFeature(features3, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions7)
        
        selectionIntentRuleOptions7.Dispose()
        section3.AllowSelfIntersection(False)
        
        section3.AllowDegenerateCurves(False)
        
        rules7 = [None] * 1 
        rules7[0] = curveFeatureRule3
        helpPoint7 = NXOpen.Point3d(0.0, 0.0, 0.0)
        section3.AddToSection(rules7, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint7, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId76, None)
        
        sections2 = [NXOpen.Section.Null] * 1 
        sections2[0] = section3
        sweptBuilder2.AlignmentMethod.SetSections(sections2)
        
        theSession.DeleteUndoMark(markId75, None)
        
        markId77 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId78 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        section3.RemoveRules(spline4, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.Section.Mode.Create)
        
        theSession.DeleteUndoMark(markId78, None)
        
        sweptBuilder2.AlignmentMethod.RemoveSectionAtIndex(0)
        
        theSession.DeleteUndoMark(markId77, None)
        
        markId79 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId80 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
        selectionIntentRuleOptions8 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions8.SetSelectedFromInactive(False)
        
        features4 = [NXOpen.Features.Feature.Null] * 1 
        features4[0] = compositeCurve1
        curveFeatureRule4 = workPart.ScRuleFactory.CreateRuleCurveFeature(features4, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions8)
        
        selectionIntentRuleOptions8.Dispose()
        section3.AllowSelfIntersection(False)
        
        section3.AllowDegenerateCurves(False)
        
        rules8 = [None] * 1 
        rules8[0] = curveFeatureRule4
        helpPoint8 = NXOpen.Point3d(0.0, 0.0, 0.0)
        section3.AddToSection(rules8, NXOpen.NXObject.Null, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint8, NXOpen.Section.Mode.Create, False)
        
        theSession.DeleteUndoMark(markId80, None)
        
        sections3 = [NXOpen.Section.Null] * 1 
        sections3[0] = section3
        sweptBuilder2.AlignmentMethod.SetSections(sections3)
        
        expression36 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
        
        theSession.DeleteUndoMark(markId79, None)
        
        section4 = workPart.Sections.CreateSection(0.00095, 0.001, 0.050000000000000003)
        
        sweptBuilder2.GuideList.Append(section4)
        
        section4.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.OnlyCurves)
        
        markId81 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "section mark")
        
        markId82 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, None)
        
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
        
        theSession.DeleteUndoMark(markId82, None)
        
        sweptBuilder2.ScalingMethod.AreaLaw.AlongSpineData.SetFeatureSpine(section4)
        
        sweptBuilder2.ScalingMethod.PerimeterLaw.AlongSpineData.SetFeatureSpine(section4)
        
        markId83 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId83, "Update Law Data", False)
        
        markId84 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId84, "Update Law Data", False)
        
        sweptBuilder2.OrientationMethod.AngularLaw.AlongSpineData.SetFeatureSpine(section4)
        
        markId85 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Update Law Data")
        
        theSession.DeleteUndoMarksUpToMark(markId85, "Update Law Data", False)
        
        theSession.DeleteUndoMark(markId81, None)
        
        markId86 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "扫掠")
        
        theSession.DeleteUndoMark(markId86, None)
        
        markId87 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "扫掠")
        
        nXObject12 = sweptBuilder2.Commit()
        
        displayModification2 = theSession.DisplayManager.NewDisplayModification()
        
        displayModification2.ApplyToAllFaces = False
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects31 = [NXOpen.DisplayableObject.Null] * 1 
        swept2 = nXObject12
        face19 = swept2.FindObject("FACE 10015 {(262.1086446359832,2.652894767263,-0.1236402380488) SWEPT(5)}")
        objects31[0] = face19
        displayModification2.Apply(objects31)
        
        face19.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects32 = [NXOpen.DisplayableObject.Null] * 1 
        face20 = swept2.FindObject("FACE 10014 {(262.1681905735861,1.7283056363762,0.8570856681183) SWEPT(5)}")
        objects32[0] = face20
        displayModification2.Apply(objects32)
        
        face20.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects33 = [NXOpen.DisplayableObject.Null] * 1 
        face21 = swept2.FindObject("FACE 10013 {(262.2102127462241,1.0470006826553,1.1584621695912) SWEPT(5)}")
        objects33[0] = face21
        displayModification2.Apply(objects33)
        
        face21.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects34 = [NXOpen.DisplayableObject.Null] * 1 
        face22 = swept2.FindObject("FACE 10012 {(262.2097265708398,1.0748973829203,1.4263888053355) SWEPT(5)}")
        objects34[0] = face22
        displayModification2.Apply(objects34)
        
        face22.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects35 = [NXOpen.DisplayableObject.Null] * 1 
        face23 = swept2.FindObject("FACE 10011 {(261.7553852964263,8.7270678484925,2.0455552986704) SWEPT(5)}")
        objects35[0] = face23
        displayModification2.Apply(objects35)
        
        face23.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects36 = [NXOpen.DisplayableObject.Null] * 1 
        face24 = swept2.FindObject("FACE 10010 {(261.2958821351687,16.3747984492826,1.4325827218921) SWEPT(5)}")
        objects36[0] = face24
        displayModification2.Apply(objects36)
        
        face24.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects37 = [NXOpen.DisplayableObject.Null] * 1 
        face25 = swept2.FindObject("FACE 10009 {(261.2923000123251,16.4158654833668,1.1762222860232) SWEPT(5)}")
        objects37[0] = face25
        displayModification2.Apply(objects37)
        
        face25.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects38 = [NXOpen.DisplayableObject.Null] * 1 
        face26 = swept2.FindObject("FACE 10008 {(261.3239361263919,15.8621941183211,0.8504292445884) SWEPT(5)}")
        objects38[0] = face26
        displayModification2.Apply(objects38)
        
        face26.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects39 = [NXOpen.DisplayableObject.Null] * 1 
        face27 = swept2.FindObject("FACE 10007 {(261.3625858626773,15.1389171971563,-0.1830357114156) SWEPT(5)}")
        objects39[0] = face27
        displayModification2.Apply(objects39)
        
        face27.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects40 = [NXOpen.DisplayableObject.Null] * 1 
        face28 = swept2.FindObject("FACE 10006 {(261.2872090901781,16.398127140349,-0.2201452141863) SWEPT(5)}")
        objects40[0] = face28
        displayModification2.Apply(objects40)
        
        face28.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects41 = [NXOpen.DisplayableObject.Null] * 1 
        face29 = swept2.FindObject("FACE 10005 {(261.2263342256081,17.4800025163761,0.6303655624164) SWEPT(5)}")
        objects41[0] = face29
        displayModification2.Apply(objects41)
        
        face29.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects42 = [NXOpen.DisplayableObject.Null] * 1 
        face30 = swept2.FindObject("FACE 10004 {(261.2494552739408,17.214932634233,2.2850609169071) SWEPT(5)}")
        objects42[0] = face30
        displayModification2.Apply(objects42)
        
        face30.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects43 = [NXOpen.DisplayableObject.Null] * 1 
        face31 = swept2.FindObject("FACE 10003 {(261.7606509529494,8.7272367116535,3.2433349308956) SWEPT(5)}")
        objects43[0] = face31
        displayModification2.Apply(objects43)
        
        face31.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects44 = [NXOpen.DisplayableObject.Null] * 1 
        face32 = swept2.FindObject("FACE 10016 {(262.1880082955782,1.3136677697039,-0.2664735861349) SWEPT(5)}")
        objects44[0] = face32
        displayModification2.Apply(objects44)
        
        face32.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects45 = [NXOpen.DisplayableObject.Null] * 1 
        face33 = swept2.FindObject("FACE 2 {(522.3445569741205,34.0158989637805,-0.3457365371509) SWEPT(5)}")
        objects45[0] = face33
        displayModification2.Apply(objects45)
        
        face33.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects46 = [NXOpen.DisplayableObject.Null] * 1 
        face34 = swept2.FindObject("FACE 10002 {(262.264933734811,0.2117301597589,2.2549991698254) SWEPT(5)}")
        objects46[0] = face34
        displayModification2.Apply(objects46)
        
        face34.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects47 = [NXOpen.DisplayableObject.Null] * 1 
        face35 = swept2.FindObject("FACE 1 {(0,1.0818839027918,1.9571319661241) SWEPT(5)}")
        objects47[0] = face35
        displayModification2.Apply(objects47)
        
        face35.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects48 = [NXOpen.DisplayableObject.Null] * 1 
        face36 = swept2.FindObject("FACE 10001 {(262.268071785919,0.0316690665659,0.5256490148081) SWEPT(5)}")
        objects48[0] = face36
        displayModification2.Apply(objects48)
        
        face36.Color = 32767
        
        theSession.DeleteUndoMark(markId87, None)
        
        theSession.SetUndoMarkName(markId74, "扫掠")
        
        sweptBuilder2.Destroy()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        workPart.Expressions.Delete(expression36)
        
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
        
        markId88 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects49 = [NXOpen.DisplayableObject.Null] * 1 
        body2 = workPart.Bodies.FindObject("SWEPT(5)")
        objects49[0] = body2
        theSession.DisplayManager.BlankObjects(objects49)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)

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
        
        admOffsetRegionBuilder1.Distance.SetFormula("0.1")
        
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
        
        markId90 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "偏置区域")
        
        theSession.DeleteUndoMark(markId90, None)
        
        markId91 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "偏置区域")
        
        nXObject13 = admOffsetRegionBuilder1.Commit()
        
        theSession.DeleteUndoMark(markId91, None)
        
        theSession.SetUndoMarkName(markId89, "偏置区域")
        
        expression37 = admOffsetRegionBuilder1.Distance
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
        features7[0] = swept2
        bodyFeatureRule1 = workPart.ScRuleFactory.CreateRuleBodyFeature(features7, False, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions12)
        
        selectionIntentRuleOptions12.Dispose()
        rules13 = [None] * 1 
        rules13[0] = bodyFeatureRule1
        scCollector6.ReplaceRules(rules13, False)
        
        booleanBuilder1.ToolBodyCollector = scCollector6
        
        targets2 = [NXOpen.TaggedObject.Null] * 1 
        targets2[0] = body1
        booleanRegionSelect1.AssignTargets(targets2)
        
        booleanBuilder1.Destroy()
        
        scCollector5.Destroy()
        
        scCollector6.Destroy()
        
        theSession.UndoToMark(markId92, None)
        
        theSession.DeleteUndoMark(markId92, None)
        
        # ----------------------------------------------
        #   菜单：插入(S)->组合(B)->合并(U)...
        # ----------------------------------------------
        markId93 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        booleanBuilder2 = workPart.Features.CreateBooleanBuilderUsingCollector(NXOpen.Features.BooleanFeature.Null)
        
        scCollector7 = booleanBuilder2.ToolBodyCollector
        
        scCollector8 = booleanBuilder2.TargetBodyCollector
        
        booleanRegionSelect2 = booleanBuilder2.BooleanRegionSelect
        
        booleanBuilder2.Tolerance = 0.001
        
        booleanBuilder2.CopyTargets = True
        
        booleanBuilder2.CopyTools = True
        
        scCollector9 = booleanBuilder2.TargetBodyCollector
        
        booleanBuilder2.Operation = NXOpen.Features.Feature.BooleanType.Unite
        
        theSession.SetUndoMarkName(markId93, "合并 对话框")
        
        scCollector10 = workPart.ScCollectors.CreateCollector()
        
        selectionIntentRuleOptions13 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions13.SetSelectedFromInactive(False)
        
        bodies2 = [NXOpen.Body.Null] * 1 
        bodies2[0] = body1
        bodyDumbRule2 = workPart.ScRuleFactory.CreateRuleBodyDumb(bodies2, True, selectionIntentRuleOptions13)
        
        selectionIntentRuleOptions13.Dispose()
        rules14 = [None] * 1 
        rules14[0] = bodyDumbRule2
        scCollector10.ReplaceRules(rules14, False)
        
        booleanBuilder2.TargetBodyCollector = scCollector10
        
        targets3 = [NXOpen.TaggedObject.Null] * 1 
        targets3[0] = body1
        booleanRegionSelect2.AssignTargets(targets3)
        
        scCollector11 = workPart.ScCollectors.CreateCollector()
        
        selectionIntentRuleOptions14 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions14.SetSelectedFromInactive(False)
        
        features8 = [NXOpen.Features.Feature.Null] * 1 
        admOffsetRegion1 = nXObject13
        features8[0] = admOffsetRegion1
        bodyFeatureRule2 = workPart.ScRuleFactory.CreateRuleBodyFeature(features8, False, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions14)
        
        selectionIntentRuleOptions14.Dispose()
        rules15 = [None] * 1 
        rules15[0] = bodyFeatureRule2
        scCollector11.ReplaceRules(rules15, False)
        
        booleanBuilder2.ToolBodyCollector = scCollector11
        
        targets4 = [NXOpen.TaggedObject.Null] * 1 
        targets4[0] = body1
        booleanRegionSelect2.AssignTargets(targets4)
        
        markId94 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "合并")
        
        theSession.DeleteUndoMark(markId94, None)
        
        markId95 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "合并")
        
        nXObject14 = booleanBuilder2.Commit()
        
        theSession.DeleteUndoMark(markId95, None)
        
        theSession.SetUndoMarkName(markId93, "合并")
        
        booleanBuilder2.Destroy()
        
        # ----------------------------------------------
        #   菜单：分析(L)  ->测量(S)...
        # ----------------------------------------------
        markId96 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        theSession.SetUndoMarkName(markId96, "测量 对话框")
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector12 = workPart.ScCollectors.CreateCollector()
        
        scCollector12.SetMultiComponent()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        selectionIntentRuleOptions15 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions15.SetSelectedFromInactive(False)
        
        features9 = [NXOpen.Features.Feature.Null] * 1 
        features9[0] = swept1
        bodyFeatureRule3 = workPart.ScRuleFactory.CreateRuleBodyFeature(features9, True, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions15)
        
        selectionIntentRuleOptions15.Dispose()
        rules16 = [None] * 1 
        rules16[0] = bodyFeatureRule3
        scCollector12.ReplaceRules(rules16, False)
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector13 = workPart.ScCollectors.CreateCollector()
        
        scCollector13.SetMultiComponent()
        
        markId97 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        markId98 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Measurement Apply")
        
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
        measureElement1 = workPart.MeasureManager.BodyElement(measureMaster1, massUnits1, 0.98999999999999999, scCollector12)
        
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
        
        markId99 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Measurement Update")
        
        nErrs10 = theSession.UpdateManager.DoUpdate(markId99)
        
        theSession.DeleteUndoMark(markId99, "Measurement Update")
        
        theSession.DeleteUndoMark(markId98, "Measurement Apply")
        
        datadeleted1 = theSession.DeleteTransientDynamicSectionCutData()
        
        theSession.DeleteUndoMark(markId97, None)
        
        theSession.SetUndoMarkName(markId96, "测量")
        
        scCollector13.Destroy()
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        markId100 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Start")
        
        theSession.SetUndoMarkName(markId100, "测量 对话框")
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector14 = workPart.ScCollectors.CreateCollector()
        
        scCollector14.SetMultiComponent()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        # ----------------------------------------------
        #   对话开始 测量
        # ----------------------------------------------
        selectionIntentRuleOptions16 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions16.SetSelectedFromInactive(False)
        
        features10 = [NXOpen.Features.Feature.Null] * 1 
        features10[0] = admOffsetRegion1
        bodyFeatureRule4 = workPart.ScRuleFactory.CreateRuleBodyFeature(features10, True, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions16)
        
        selectionIntentRuleOptions16.Dispose()
        rules17 = [None] * 1 
        rules17[0] = bodyFeatureRule4
        scCollector14.ReplaceRules(rules17, False)
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector15 = workPart.ScCollectors.CreateCollector()
        
        scCollector15.SetMultiComponent()
        
        markId101 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        markId102 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Measurement Apply")
        
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
        measureElement6 = workPart.MeasureManager.BodyElement(measureMaster2, massUnits2, 0.98999999999999999, scCollector14)
        
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
        
        markId103 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Measurement Update")
        
        nErrs11 = theSession.UpdateManager.DoUpdate(markId103)
        
        theSession.DeleteUndoMark(markId103, "Measurement Update")
        
        theSession.DeleteUndoMark(markId102, "Measurement Apply")
        
        datadeleted2 = theSession.DeleteTransientDynamicSectionCutData()
        
        theSession.DeleteUndoMark(markId101, None)
        
        theSession.SetUndoMarkName(markId100, "测量")
        
        scCollector15.Destroy()
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        markId104 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Start")
        
        theSession.SetUndoMarkName(markId104, "测量 对话框")
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector16 = workPart.ScCollectors.CreateCollector()
        
        scCollector16.SetMultiComponent()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        # ----------------------------------------------
        #   对话开始 测量
        # ----------------------------------------------
        selectionIntentRuleOptions17 = workPart.ScRuleFactory.CreateRuleOptions()
        
        selectionIntentRuleOptions17.SetSelectedFromInactive(False)
        
        features11 = [NXOpen.Features.Feature.Null] * 1 
        booleanFeature1 = nXObject14
        features11[0] = booleanFeature1
        bodyFeatureRule5 = workPart.ScRuleFactory.CreateRuleBodyFeature(features11, True, NXOpen.DisplayableObject.Null, selectionIntentRuleOptions17)
        
        selectionIntentRuleOptions17.Dispose()
        rules18 = [None] * 1 
        rules18[0] = bodyFeatureRule5
        scCollector16.ReplaceRules(rules18, False)
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector17 = workPart.ScCollectors.CreateCollector()
        
        scCollector17.SetMultiComponent()
        
        markId105 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        theSession.DeleteUndoMark(markId105, None)
        
        markId106 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        markId107 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Measurement Apply")
        
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
        measureElement11 = workPart.MeasureManager.BodyElement(measureMaster3, massUnits3, 0.98999999999999999, scCollector16)
        
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
        
        markId108 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Measurement Update")
        
        nErrs12 = theSession.UpdateManager.DoUpdate(markId108)
        
        theSession.DeleteUndoMark(markId108, "Measurement Update")
        
        theSession.DeleteUndoMark(markId107, "Measurement Apply")
        
        datadeleted3 = theSession.DeleteTransientDynamicSectionCutData()
        
        theSession.DeleteUndoMark(markId106, None)
        
        theSession.SetUndoMarkName(markId104, "测量")
        
        scCollector17.Destroy()
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        markId109 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects50 = [NXOpen.DisplayableObject.Null] * 1 
        body3 = workPart.Bodies.FindObject("UNITE(7)")
        objects50[0] = body3
        theSession.DisplayManager.BlankObjects(objects50)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        markId110 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects51 = [NXOpen.DisplayableObject.Null] * 1 
        generalLabel1 = annotation1
        objects51[0] = generalLabel1
        theSession.DisplayManager.BlankObjects(objects51)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        markId111 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects52 = [NXOpen.DisplayableObject.Null] * 1 
        generalLabel2 = annotation2
        objects52[0] = generalLabel2
        theSession.DisplayManager.BlankObjects(objects52)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        markId112 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects53 = [NXOpen.DisplayableObject.Null] * 1 
        generalLabel3 = annotation3
        objects53[0] = generalLabel3
        theSession.DisplayManager.BlankObjects(objects53)

        springback_strip_volume = float(annotation1.GetText()[0].split()[1])
        print(springback_strip_volume)

        prediction_strip_volume = float(annotation2.GetText()[0].split()[1])
        print(prediction_strip_volume)

        union_strip_volume = float(annotation3.GetText()[0].split()[1])
        print(union_strip_volume)

        intersection_strip_volume = springback_strip_volume + prediction_strip_volume - union_strip_volume
        print(intersection_strip_volume)

        iou_3d = intersection_strip_volume / union_strip_volume
        print(iou_3d)
        
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
        datumCsys1 = workPart.Features.FindObject("DATUM_CSYS(0)")
        point1 = datumCsys1.FindObject("POINT 1")
        objects1[1] = point1
        cartesianCoordinateSystem1 = datumCsys1.FindObject("CSYSTEM 1")
        objects1[2] = cartesianCoordinateSystem1
        datumPlane1 = workPart.Datums.FindObject("DATUM_CSYS(0) YZ plane")
        objects1[3] = datumPlane1
        datumAxis2 = workPart.Datums.FindObject("DATUM_CSYS(0) Z axis")
        objects1[4] = datumAxis2
        datumAxis3 = workPart.Datums.FindObject("DATUM_CSYS(0) X axis")
        objects1[5] = datumAxis3
        datumPlane2 = workPart.Datums.FindObject("DATUM_CSYS(0) XZ plane")
        objects1[6] = datumPlane2
        datumPlane3 = workPart.Datums.FindObject("DATUM_CSYS(0) XY plane")
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
        
        step214Importer1.OutputFile = "G:\\PHD2\\Code\\e2e_ablation\\scripts\\iou_3d\\iou_3d_strip_section_type_2.prt"
        
        step214Importer1.SettingsFile = "D:\\NX1953\\NX1953\\step214ug\\step214ug.def"
        
        step214Importer1.ObjectTypes.ProductData = True
        
        step214Importer1.OutputFile = ""
        
        theSession.SetUndoMarkName(markId5, "导入 STEP214 文件 对话框")
        
        step214Importer1.SetMode(NXOpen.BaseImporter.Mode.NativeFileSystem)
        
        step214Importer1.OutputFile = "G:\\PHD2\\Code\\e2e_ablation\\scripts\\iou_3d\\iou_3d_strip_section_type_2.prt"
        
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
        #   菜单：工具(T)->重复命令(R)->5 抽取几何特征
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
        
        curves1 = [NXOpen.IBaseCurve.Null] * 16 
        line1 = workPart.Lines.FindObject("ENTITY 3 4 1")
        curves1[0] = line1
        line2 = workPart.Lines.FindObject("ENTITY 3 3 1")
        curves1[1] = line2
        line3 = workPart.Lines.FindObject("ENTITY 3 2 1")
        curves1[2] = line3
        line4 = workPart.Lines.FindObject("ENTITY 3 1 1")
        curves1[3] = line4
        arc1 = workPart.Arcs.FindObject("ENTITY 5 12 1")
        curves1[4] = arc1
        arc2 = workPart.Arcs.FindObject("ENTITY 5 11 1")
        curves1[5] = arc2
        arc3 = workPart.Arcs.FindObject("ENTITY 5 10 1")
        curves1[6] = arc3
        arc4 = workPart.Arcs.FindObject("ENTITY 5 9 1")
        curves1[7] = arc4
        arc5 = workPart.Arcs.FindObject("ENTITY 5 8 1")
        curves1[8] = arc5
        arc6 = workPart.Arcs.FindObject("ENTITY 5 7 1")
        curves1[9] = arc6
        arc7 = workPart.Arcs.FindObject("ENTITY 5 6 1")
        curves1[10] = arc7
        arc8 = workPart.Arcs.FindObject("ENTITY 5 5 1")
        curves1[11] = arc8
        arc9 = workPart.Arcs.FindObject("ENTITY 5 4 1")
        curves1[12] = arc9
        arc10 = workPart.Arcs.FindObject("ENTITY 5 3 1")
        curves1[13] = arc10
        arc11 = workPart.Arcs.FindObject("ENTITY 5 2 1")
        curves1[14] = arc11
        arc12 = workPart.Arcs.FindObject("ENTITY 5 1 1")
        curves1[15] = arc12
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
        
        objects2 = [NXOpen.DisplayableObject.Null] * 16 
        compositeCurve1 = nXObject3
        arc13 = compositeCurve1.FindObject("CURVE 15 {5 (-1.0409002510708,-7.6352891479658,0)}")
        objects2[0] = arc13
        line5 = compositeCurve1.FindObject("CURVE 4 {3 (-0.2931128405779,8.7604942913567,0)}")
        objects2[1] = line5
        arc14 = compositeCurve1.FindObject("CURVE 11 {5 (-1.6735967925619,-6.6182341718042,0)}")
        objects2[2] = arc14
        arc15 = compositeCurve1.FindObject("CURVE 7 {5 (-1.6894926617046,8.7780670921438,0)}")
        objects2[3] = arc15
        line6 = compositeCurve1.FindObject("CURVE 14 {3 (-0.248613917961,-6.3508897414195,0)}")
        objects2[4] = line6
        arc16 = compositeCurve1.FindObject("CURVE 3 {5 (-1.1435011876657,9.8441810862489,0)}")
        objects2[5] = arc16
        arc17 = compositeCurve1.FindObject("CURVE 10 {5 (-1.9415228210483,-6.5903255056356,0)}")
        objects2[6] = arc17
        line7 = compositeCurve1.FindObject("CURVE 6 {3 (-1.3637634106659,8.2234532362034,0)}")
        objects2[7] = line7
        arc18 = compositeCurve1.FindObject("CURVE 13 {5 (-0.3912862818355,-5.009291912854,0)}")
        objects2[8] = arc18
        arc19 = compositeCurve1.FindObject("CURVE 2 {5 (-2.7982458486376,9.578404262982,-0)}")
        objects2[9] = arc19
        arc20 = compositeCurve1.FindObject("CURVE 9 {5 (-2.5597674298829,1.0754194004682,0)}")
        objects2[10] = arc20
        arc21 = compositeCurve1.FindObject("CURVE 16 {5 (-2.7702464865025,-7.4551508200582,0)}")
        objects2[11] = arc21
        arc22 = compositeCurve1.FindObject("CURVE 5 {5 (-0.3303754855487,7.4990309628211,0)}")
        objects2[12] = arc22
        line8 = compositeCurve1.FindObject("CURVE 12 {3 (-1.3721345168131,-5.9356702788398,0)}")
        objects2[13] = line8
        arc23 = compositeCurve1.FindObject("CURVE 1 {5 (-3.7575594561337,1.0754194004682,0)}")
        objects2[14] = arc23
        arc24 = compositeCurve1.FindObject("CURVE 8 {5 (-1.9458607365815,8.7368905323186,0)}")
        objects2[15] = arc24
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
        
        objects3 = [NXOpen.TaggedObject.Null] * 16 
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
        
        objects4 = [NXOpen.NXObject.Null] * 16 
        objects4[0] = arc23
        objects4[1] = arc19
        objects4[2] = arc16
        objects4[3] = line5
        objects4[4] = arc22
        objects4[5] = line7
        objects4[6] = arc15
        objects4[7] = arc24
        objects4[8] = arc20
        objects4[9] = arc17
        objects4[10] = arc14
        objects4[11] = line8
        objects4[12] = arc18
        objects4[13] = line6
        objects4[14] = arc13
        objects4[15] = arc21
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
        #   菜单：工具(T)->重复命令(R)->8 拟合曲线
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
        
        scaleAboutPoint1 = NXOpen.Point3d(-53.829806973872607, -5.4892237374672828, 0.0)
        viewCenter1 = NXOpen.Point3d(53.829806973872607, 5.4892237374672828, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint1, viewCenter1)
        
        scaleAboutPoint2 = NXOpen.Point3d(-33.997772825603775, -8.9244153667209982, 0.0)
        viewCenter2 = NXOpen.Point3d(33.997772825603775, 8.9244153667209982, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint2, viewCenter2)
        
        scaleAboutPoint3 = NXOpen.Point3d(-13.599109130241484, -8.7260950252383012, 0.0)
        viewCenter3 = NXOpen.Point3d(13.59910913024151, 8.7260950252383012, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint3, viewCenter3)
        
        scaleAboutPoint4 = NXOpen.Point3d(-3.0824647361880726, -7.7061618404701919, 0.0)
        viewCenter4 = NXOpen.Point3d(3.0824647361880726, 7.7061618404701919, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint4, viewCenter4)
        
        scaleAboutPoint5 = NXOpen.Point3d(3.1912576092300067, -6.3099866364320603, 0.0)
        viewCenter5 = NXOpen.Point3d(-3.1912576092300067, 6.3099866364320603, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint5, viewCenter5)
        
        scaleAboutPoint6 = NXOpen.Point3d(4.6418292497891258, -4.815897846656199, 0.0)
        viewCenter6 = NXOpen.Point3d(-4.6418292497891258, 4.815897846656199, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint6, viewCenter6)
        
        scaleAboutPoint7 = NXOpen.Point3d(6.8699072896878866, -2.8315158423713567, 0.0)
        viewCenter7 = NXOpen.Point3d(-6.8699072896878643, 2.8315158423713567, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint7, viewCenter7)
        
        scaleAboutPoint8 = NXOpen.Point3d(9.7292741075579663, -0.55701950997468896, 0.0)
        viewCenter8 = NXOpen.Point3d(-9.7292741075579485, 0.55701950997468896, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint8, viewCenter8)
        
        scaleAboutPoint9 = NXOpen.Point3d(28.281737253115075, 10.724482298712742, 0.0)
        viewCenter9 = NXOpen.Point3d(-28.281737253115033, -10.724482298712742, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint9, viewCenter9)
        
        scaleAboutPoint10 = NXOpen.Point3d(22.625389802492059, 8.5795858389701927, 0.0)
        viewCenter10 = NXOpen.Point3d(-22.625389802492002, -8.579585838970198, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint10, viewCenter10)
        
        scaleAboutPoint11 = NXOpen.Point3d(18.100311841993662, 6.8636686711761499, 0.0)
        viewCenter11 = NXOpen.Point3d(-18.100311841993605, -6.8636686711761588, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint11, viewCenter11)
        
        scaleAboutPoint12 = NXOpen.Point3d(14.480249473594933, 5.4909349369409197, 0.0)
        viewCenter12 = NXOpen.Point3d(-14.480249473594876, -5.4909349369409268, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint12, viewCenter12)
        
        scaleAboutPoint13 = NXOpen.Point3d(11.559863025138814, 4.3927479495527386, 0.0)
        viewCenter13 = NXOpen.Point3d(-11.559863025138752, -4.3927479495527413, 0.0)
        workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint13, viewCenter13)
        
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
        helpPoint2 = NXOpen.Point3d(1.0409512194617889, 0.0056995333139508164, -0.00058180114037642908)
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
        objects8[0] = point60
        objects8[1] = point61
        objects8[2] = point62
        objects8[3] = point63
        objects8[4] = point64
        objects8[5] = point65
        objects8[6] = point66
        objects8[7] = point67
        objects8[8] = point68
        objects8[9] = point69
        objects8[10] = point70
        objects8[11] = point71
        objects8[12] = spline1
        objects8[13] = point213
        objects8[14] = point214
        objects8[15] = point215
        objects8[16] = point216
        objects8[17] = point217
        objects8[18] = point218
        objects8[19] = point219
        objects8[20] = point220
        objects8[21] = point221
        objects8[22] = point222
        objects8[23] = point223
        objects8[24] = point224
        objects8[25] = point225
        objects8[26] = point226
        objects8[27] = point227
        objects8[28] = point228
        objects8[29] = point229
        objects8[30] = point230
        objects8[31] = point231
        objects8[32] = point232
        objects8[33] = point233
        objects8[34] = point234
        objects8[35] = point235
        objects8[36] = point236
        objects8[37] = point237
        objects8[38] = point238
        objects8[39] = point108
        objects8[40] = point109
        objects8[41] = point110
        objects8[42] = point111
        objects8[43] = point112
        objects8[44] = point113
        objects8[45] = point114
        objects8[46] = point115
        objects8[47] = point116
        objects8[48] = point117
        objects8[49] = point118
        objects8[50] = point119
        objects8[51] = point120
        objects8[52] = point121
        objects8[53] = point122
        objects8[54] = point123
        objects8[55] = point124
        objects8[56] = point125
        objects8[57] = point126
        objects8[58] = point127
        objects8[59] = point128
        objects8[60] = point129
        objects8[61] = point130
        objects8[62] = point131
        objects8[63] = point132
        objects8[64] = point133
        objects8[65] = point134
        objects8[66] = point135
        objects8[67] = point136
        objects8[68] = point137
        objects8[69] = point138
        objects8[70] = point139
        objects8[71] = point140
        objects8[72] = point177
        objects8[73] = point178
        objects8[74] = point179
        objects8[75] = point180
        objects8[76] = point181
        objects8[77] = point182
        objects8[78] = point183
        objects8[79] = point184
        objects8[80] = point185
        objects8[81] = point186
        objects8[82] = point187
        objects8[83] = point188
        objects8[84] = point189
        objects8[85] = point190
        objects8[86] = point191
        objects8[87] = point192
        objects8[88] = point193
        objects8[89] = point194
        objects8[90] = point195
        objects8[91] = point196
        objects8[92] = point197
        objects8[93] = point198
        objects8[94] = point199
        objects8[95] = point200
        objects8[96] = point201
        objects8[97] = point202
        objects8[98] = point203
        objects8[99] = point204
        objects8[100] = point205
        objects8[101] = point206
        objects8[102] = point207
        objects8[103] = point208
        objects8[104] = point209
        objects8[105] = point210
        objects8[106] = point211
        objects8[107] = point212
        objects8[108] = point56
        objects8[109] = point57
        objects8[110] = point58
        objects8[111] = point59
        objects8[112] = point6
        objects8[113] = point7
        objects8[114] = point8
        objects8[115] = point9
        objects8[116] = point10
        objects8[117] = point11
        objects8[118] = point12
        objects8[119] = point13
        objects8[120] = point14
        objects8[121] = point15
        objects8[122] = point16
        objects8[123] = point17
        objects8[124] = point18
        objects8[125] = point19
        objects8[126] = point20
        objects8[127] = point21
        objects8[128] = point22
        objects8[129] = point23
        objects8[130] = point24
        objects8[131] = point25
        objects8[132] = point26
        objects8[133] = point27
        objects8[134] = point28
        objects8[135] = point29
        objects8[136] = point30
        objects8[137] = point31
        objects8[138] = point32
        objects8[139] = point33
        objects8[140] = point34
        objects8[141] = point35
        objects8[142] = point36
        objects8[143] = point37
        objects8[144] = point38
        objects8[145] = point39
        objects8[146] = point40
        objects8[147] = point41
        objects8[148] = point42
        objects8[149] = point43
        objects8[150] = point44
        objects8[151] = point45
        objects8[152] = point46
        objects8[153] = point47
        objects8[154] = point48
        objects8[155] = point49
        objects8[156] = point50
        objects8[157] = point51
        objects8[158] = point52
        objects8[159] = point53
        objects8[160] = point54
        objects8[161] = point55
        objects8[162] = point80
        objects8[163] = point81
        objects8[164] = point82
        objects8[165] = point83
        objects8[166] = point84
        objects8[167] = point85
        objects8[168] = point86
        objects8[169] = point87
        objects8[170] = point88
        objects8[171] = point89
        objects8[172] = point90
        objects8[173] = point91
        objects8[174] = point92
        objects8[175] = point93
        objects8[176] = point94
        objects8[177] = point95
        objects8[178] = point96
        objects8[179] = point97
        objects8[180] = point98
        objects8[181] = point99
        objects8[182] = point100
        objects8[183] = point101
        objects8[184] = point102
        objects8[185] = point103
        objects8[186] = point104
        objects8[187] = point105
        objects8[188] = point106
        objects8[189] = point107
        objects8[190] = point300
        objects8[191] = point301
        objects8[192] = point302
        objects8[193] = point303
        objects8[194] = point304
        objects8[195] = point305
        objects8[196] = point306
        objects8[197] = group1
        objects8[198] = point72
        objects8[199] = point73
        objects8[200] = point74
        objects8[201] = point75
        objects8[202] = point76
        objects8[203] = point77
        objects8[204] = point78
        objects8[205] = point79
        objects8[206] = point239
        objects8[207] = point240
        objects8[208] = point241
        objects8[209] = point242
        objects8[210] = point243
        objects8[211] = point244
        objects8[212] = point245
        objects8[213] = point246
        objects8[214] = point247
        objects8[215] = point248
        objects8[216] = point249
        objects8[217] = point250
        objects8[218] = point251
        objects8[219] = point252
        objects8[220] = point253
        objects8[221] = point254
        objects8[222] = point255
        objects8[223] = point256
        objects8[224] = point257
        objects8[225] = point258
        objects8[226] = point259
        objects8[227] = point260
        objects8[228] = point261
        objects8[229] = point262
        objects8[230] = point263
        objects8[231] = point264
        objects8[232] = point265
        objects8[233] = point266
        objects8[234] = point267
        objects8[235] = point268
        objects8[236] = point269
        objects8[237] = point270
        objects8[238] = point271
        objects8[239] = point272
        objects8[240] = point273
        objects8[241] = point274
        objects8[242] = point275
        objects8[243] = point276
        objects8[244] = point277
        objects8[245] = point278
        objects8[246] = point279
        objects8[247] = point280
        objects8[248] = point281
        objects8[249] = point282
        objects8[250] = point283
        objects8[251] = point284
        objects8[252] = point285
        objects8[253] = point286
        objects8[254] = point287
        objects8[255] = point288
        objects8[256] = point289
        objects8[257] = point290
        objects8[258] = point291
        objects8[259] = point292
        objects8[260] = point293
        objects8[261] = point294
        objects8[262] = point295
        objects8[263] = point296
        objects8[264] = point297
        objects8[265] = point298
        objects8[266] = point299
        objects8[267] = point141
        objects8[268] = point142
        objects8[269] = point143
        objects8[270] = point144
        objects8[271] = point145
        objects8[272] = point146
        objects8[273] = point147
        objects8[274] = point148
        objects8[275] = point149
        objects8[276] = point150
        objects8[277] = point151
        objects8[278] = point152
        objects8[279] = point153
        objects8[280] = point154
        objects8[281] = point155
        objects8[282] = point156
        objects8[283] = point157
        objects8[284] = point158
        objects8[285] = point159
        objects8[286] = point160
        objects8[287] = point161
        objects8[288] = point162
        objects8[289] = point163
        objects8[290] = point164
        objects8[291] = point165
        objects8[292] = point166
        objects8[293] = point167
        objects8[294] = point168
        objects8[295] = point169
        objects8[296] = point170
        objects8[297] = point171
        objects8[298] = point172
        objects8[299] = point173
        objects8[300] = point174
        objects8[301] = point175
        objects8[302] = point176
        nErrs5 = theSession.UpdateManager.AddObjectsToDeleteList(objects8)
        
        id2 = theSession.NewestVisibleUndoMark
        
        nErrs6 = theSession.UpdateManager.DoUpdate(id2)
        
        theSession.DeleteUndoMark(markId41, None)
        
        # ----------------------------------------------
        #   菜单：工具(T)->重复命令(R)->8 扫掠
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
        face1 = swept1.FindObject("FACE 10015 {(262.2915677130831,2.6581970204865,-0.2262521988874) SWEPT(3)}")
        objects9[0] = face1
        displayModification1.Apply(objects9)
        
        face1.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects10 = [NXOpen.DisplayableObject.Null] * 1 
        face2 = swept1.FindObject("FACE 10014 {(262.3514985677187,1.7336082979685,0.7544507743374) SWEPT(3)}")
        objects10[0] = face2
        displayModification1.Apply(objects10)
        
        face2.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects11 = [NXOpen.DisplayableObject.Null] * 1 
        face3 = swept1.FindObject("FACE 10013 {(262.3936744800214,1.0523054818288,1.0558108604208) SWEPT(3)}")
        objects11[0] = face3
        displayModification1.Apply(objects11)
        
        face3.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects12 = [NXOpen.DisplayableObject.Null] * 1 
        face4 = swept1.FindObject("FACE 10012 {(262.3932684227858,1.080200872723,1.3237378468056) SWEPT(3)}")
        objects12[0] = face4
        displayModification1.Apply(objects12)
        
        face4.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects13 = [NXOpen.DisplayableObject.Null] * 1 
        face5 = swept1.FindObject("FACE 10011 {(261.9384349686234,8.7323298734859,1.943084118588) SWEPT(3)}")
        objects13[0] = face5
        displayModification1.Apply(objects13)
        
        face5.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects14 = [NXOpen.DisplayableObject.Null] * 1 
        face6 = swept1.FindObject("FACE 10010 {(261.4780601250592,16.3800244035669,1.330292628965) SWEPT(3)}")
        objects14[0] = face6
        displayModification1.Apply(objects14)
        
        face6.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects15 = [NXOpen.DisplayableObject.Null] * 1 
        face7 = swept1.FindObject("FACE 10009 {(261.4743952940463,16.4210923472313,1.0739334558078) SWEPT(3)}")
        objects15[0] = face7
        displayModification1.Apply(objects15)
        
        face7.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects16 = [NXOpen.DisplayableObject.Null] * 1 
        face8 = swept1.FindObject("FACE 10008 {(261.5059803824963,15.867425207377,0.7481277288743) SWEPT(3)}")
        objects16[0] = face8
        displayModification1.Apply(objects16)
        
        face8.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects17 = [NXOpen.DisplayableObject.Null] * 1 
        face9 = swept1.FindObject("FACE 10007 {(261.5443760405748,15.1441564557686,-0.2853531012528) SWEPT(3)}")
        objects17[0] = face9
        displayModification1.Apply(objects17)
        
        face9.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects18 = [NXOpen.DisplayableObject.Null] * 1 
        face10 = swept1.FindObject("FACE 10006 {(261.4688754207743,16.4033601816499,-0.3224328609676) SWEPT(3)}")
        objects18[0] = face10
        displayModification1.Apply(objects18)
        
        face10.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects19 = [NXOpen.DisplayableObject.Null] * 1 
        face11 = swept1.FindObject("FACE 10005 {(261.4081662142182,17.4852263690825,0.5281024577509) SWEPT(3)}")
        objects19[0] = face11
        displayModification1.Apply(objects19)
        
        face11.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects20 = [NXOpen.DisplayableObject.Null] * 1 
        face12 = swept1.FindObject("FACE 10004 {(261.431821107889,17.2201506159942,2.1827896619852) SWEPT(3)}")
        objects20[0] = face12
        displayModification1.Apply(objects20)
        
        face12.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects21 = [NXOpen.DisplayableObject.Null] * 1 
        face13 = swept1.FindObject("FACE 10003 {(261.9440699150994,8.7324935139761,3.1408623807818) SWEPT(3)}")
        objects21[0] = face13
        displayModification1.Apply(objects21)
        
        face13.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects22 = [NXOpen.DisplayableObject.Null] * 1 
        face14 = swept1.FindObject("FACE 10016 {(262.3710068820159,1.3189774300761,-0.3691169709414) SWEPT(3)}")
        objects22[0] = face14
        displayModification1.Apply(objects22)
        
        face14.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects23 = [NXOpen.DisplayableObject.Null] * 1 
        face15 = swept1.FindObject("FACE 2 {(522.7086093703734,34.0253243557089,-0.1390083222137) SWEPT(3)}")
        objects23[0] = face15
        displayModification1.Apply(objects23)
        
        face15.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects24 = [NXOpen.DisplayableObject.Null] * 1 
        face16 = swept1.FindObject("FACE 10002 {(262.448808119882,0.21703440993,2.1523269015845) SWEPT(3)}")
        objects24[0] = face16
        displayModification1.Apply(objects24)
        
        face16.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects25 = [NXOpen.DisplayableObject.Null] * 1 
        face17 = swept1.FindObject("FACE 1 {(0,1.0818839027918,1.9571319661241) SWEPT(3)}")
        objects25[0] = face17
        displayModification1.Apply(objects25)
        
        face17.Color = 32767
        
        displayModification1.SetNewGrid(0, 0)
        
        displayModification1.PoleDisplayState = False
        
        displayModification1.KnotDisplayState = False
        
        objects26 = [NXOpen.DisplayableObject.Null] * 1 
        face18 = swept1.FindObject("FACE 10001 {(262.4514290431105,0.0369817681565,0.4229744833347) SWEPT(3)}")
        objects26[0] = face18
        displayModification1.Apply(objects26)
        
        face18.Color = 32767
        
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
        
        objects27 = [NXOpen.DisplayableObject.Null] * 1 
        body1 = workPart.Bodies.FindObject("SWEPT(3)")
        objects27[0] = body1
        theSession.DisplayManager.BlankObjects(objects27)
        
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
        objects28 = [NXOpen.TaggedObject.Null] * 302 
        point309 = workPart.Points.FindObject("ENTITY 2 301 1")
        objects28[0] = point309
        point310 = workPart.Points.FindObject("ENTITY 2 300 1")
        objects28[1] = point310
        point311 = workPart.Points.FindObject("ENTITY 2 299 1")
        objects28[2] = point311
        point312 = workPart.Points.FindObject("ENTITY 2 298 1")
        objects28[3] = point312
        point313 = workPart.Points.FindObject("ENTITY 2 297 1")
        objects28[4] = point313
        point314 = workPart.Points.FindObject("ENTITY 2 296 1")
        objects28[5] = point314
        point315 = workPart.Points.FindObject("ENTITY 2 295 1")
        objects28[6] = point315
        point316 = workPart.Points.FindObject("ENTITY 2 294 1")
        objects28[7] = point316
        point317 = workPart.Points.FindObject("ENTITY 2 293 1")
        objects28[8] = point317
        point318 = workPart.Points.FindObject("ENTITY 2 292 1")
        objects28[9] = point318
        point319 = workPart.Points.FindObject("ENTITY 2 291 1")
        objects28[10] = point319
        point320 = workPart.Points.FindObject("ENTITY 2 290 1")
        objects28[11] = point320
        point321 = workPart.Points.FindObject("ENTITY 2 289 1")
        objects28[12] = point321
        point322 = workPart.Points.FindObject("ENTITY 2 288 1")
        objects28[13] = point322
        point323 = workPart.Points.FindObject("ENTITY 2 287 1")
        objects28[14] = point323
        point324 = workPart.Points.FindObject("ENTITY 2 286 1")
        objects28[15] = point324
        point325 = workPart.Points.FindObject("ENTITY 2 285 1")
        objects28[16] = point325
        point326 = workPart.Points.FindObject("ENTITY 2 284 1")
        objects28[17] = point326
        point327 = workPart.Points.FindObject("ENTITY 2 283 1")
        objects28[18] = point327
        point328 = workPart.Points.FindObject("ENTITY 2 282 1")
        objects28[19] = point328
        point329 = workPart.Points.FindObject("ENTITY 2 281 1")
        objects28[20] = point329
        point330 = workPart.Points.FindObject("ENTITY 2 280 1")
        objects28[21] = point330
        point331 = workPart.Points.FindObject("ENTITY 2 279 1")
        objects28[22] = point331
        point332 = workPart.Points.FindObject("ENTITY 2 278 1")
        objects28[23] = point332
        point333 = workPart.Points.FindObject("ENTITY 2 277 1")
        objects28[24] = point333
        point334 = workPart.Points.FindObject("ENTITY 2 276 1")
        objects28[25] = point334
        point335 = workPart.Points.FindObject("ENTITY 2 275 1")
        objects28[26] = point335
        point336 = workPart.Points.FindObject("ENTITY 2 274 1")
        objects28[27] = point336
        point337 = workPart.Points.FindObject("ENTITY 2 273 1")
        objects28[28] = point337
        point338 = workPart.Points.FindObject("ENTITY 2 272 1")
        objects28[29] = point338
        point339 = workPart.Points.FindObject("ENTITY 2 271 1")
        objects28[30] = point339
        point340 = workPart.Points.FindObject("ENTITY 2 270 1")
        objects28[31] = point340
        point341 = workPart.Points.FindObject("ENTITY 2 269 1")
        objects28[32] = point341
        point342 = workPart.Points.FindObject("ENTITY 2 268 1")
        objects28[33] = point342
        point343 = workPart.Points.FindObject("ENTITY 2 267 1")
        objects28[34] = point343
        point344 = workPart.Points.FindObject("ENTITY 2 266 1")
        objects28[35] = point344
        point345 = workPart.Points.FindObject("ENTITY 2 265 1")
        objects28[36] = point345
        point346 = workPart.Points.FindObject("ENTITY 2 264 1")
        objects28[37] = point346
        point347 = workPart.Points.FindObject("ENTITY 2 263 1")
        objects28[38] = point347
        point348 = workPart.Points.FindObject("ENTITY 2 262 1")
        objects28[39] = point348
        point349 = workPart.Points.FindObject("ENTITY 2 261 1")
        objects28[40] = point349
        point350 = workPart.Points.FindObject("ENTITY 2 260 1")
        objects28[41] = point350
        point351 = workPart.Points.FindObject("ENTITY 2 259 1")
        objects28[42] = point351
        point352 = workPart.Points.FindObject("ENTITY 2 258 1")
        objects28[43] = point352
        point353 = workPart.Points.FindObject("ENTITY 2 257 1")
        objects28[44] = point353
        point354 = workPart.Points.FindObject("ENTITY 2 256 1")
        objects28[45] = point354
        point355 = workPart.Points.FindObject("ENTITY 2 255 1")
        objects28[46] = point355
        point356 = workPart.Points.FindObject("ENTITY 2 254 1")
        objects28[47] = point356
        point357 = workPart.Points.FindObject("ENTITY 2 253 1")
        objects28[48] = point357
        point358 = workPart.Points.FindObject("ENTITY 2 252 1")
        objects28[49] = point358
        point359 = workPart.Points.FindObject("ENTITY 2 251 1")
        objects28[50] = point359
        point360 = workPart.Points.FindObject("ENTITY 2 250 1")
        objects28[51] = point360
        point361 = workPart.Points.FindObject("ENTITY 2 249 1")
        objects28[52] = point361
        point362 = workPart.Points.FindObject("ENTITY 2 248 1")
        objects28[53] = point362
        point363 = workPart.Points.FindObject("ENTITY 2 247 1")
        objects28[54] = point363
        point364 = workPart.Points.FindObject("ENTITY 2 246 1")
        objects28[55] = point364
        point365 = workPart.Points.FindObject("ENTITY 2 245 1")
        objects28[56] = point365
        point366 = workPart.Points.FindObject("ENTITY 2 244 1")
        objects28[57] = point366
        point367 = workPart.Points.FindObject("ENTITY 2 243 1")
        objects28[58] = point367
        point368 = workPart.Points.FindObject("ENTITY 2 242 1")
        objects28[59] = point368
        point369 = workPart.Points.FindObject("ENTITY 2 241 1")
        objects28[60] = point369
        point370 = workPart.Points.FindObject("ENTITY 2 240 1")
        objects28[61] = point370
        point371 = workPart.Points.FindObject("ENTITY 2 239 1")
        objects28[62] = point371
        point372 = workPart.Points.FindObject("ENTITY 2 238 1")
        objects28[63] = point372
        point373 = workPart.Points.FindObject("ENTITY 2 237 1")
        objects28[64] = point373
        point374 = workPart.Points.FindObject("ENTITY 2 236 1")
        objects28[65] = point374
        point375 = workPart.Points.FindObject("ENTITY 2 235 1")
        objects28[66] = point375
        point376 = workPart.Points.FindObject("ENTITY 2 234 1")
        objects28[67] = point376
        point377 = workPart.Points.FindObject("ENTITY 2 233 1")
        objects28[68] = point377
        point378 = workPart.Points.FindObject("ENTITY 2 232 1")
        objects28[69] = point378
        point379 = workPart.Points.FindObject("ENTITY 2 231 1")
        objects28[70] = point379
        point380 = workPart.Points.FindObject("ENTITY 2 230 1")
        objects28[71] = point380
        point381 = workPart.Points.FindObject("ENTITY 2 229 1")
        objects28[72] = point381
        point382 = workPart.Points.FindObject("ENTITY 2 228 1")
        objects28[73] = point382
        point383 = workPart.Points.FindObject("ENTITY 2 227 1")
        objects28[74] = point383
        point384 = workPart.Points.FindObject("ENTITY 2 226 1")
        objects28[75] = point384
        point385 = workPart.Points.FindObject("ENTITY 2 225 1")
        objects28[76] = point385
        point386 = workPart.Points.FindObject("ENTITY 2 224 1")
        objects28[77] = point386
        point387 = workPart.Points.FindObject("ENTITY 2 223 1")
        objects28[78] = point387
        point388 = workPart.Points.FindObject("ENTITY 2 222 1")
        objects28[79] = point388
        point389 = workPart.Points.FindObject("ENTITY 2 221 1")
        objects28[80] = point389
        point390 = workPart.Points.FindObject("ENTITY 2 220 1")
        objects28[81] = point390
        point391 = workPart.Points.FindObject("ENTITY 2 219 1")
        objects28[82] = point391
        point392 = workPart.Points.FindObject("ENTITY 2 218 1")
        objects28[83] = point392
        point393 = workPart.Points.FindObject("ENTITY 2 217 1")
        objects28[84] = point393
        point394 = workPart.Points.FindObject("ENTITY 2 216 1")
        objects28[85] = point394
        point395 = workPart.Points.FindObject("ENTITY 2 215 1")
        objects28[86] = point395
        point396 = workPart.Points.FindObject("ENTITY 2 214 1")
        objects28[87] = point396
        point397 = workPart.Points.FindObject("ENTITY 2 213 1")
        objects28[88] = point397
        point398 = workPart.Points.FindObject("ENTITY 2 212 1")
        objects28[89] = point398
        point399 = workPart.Points.FindObject("ENTITY 2 211 1")
        objects28[90] = point399
        point400 = workPart.Points.FindObject("ENTITY 2 210 1")
        objects28[91] = point400
        point401 = workPart.Points.FindObject("ENTITY 2 209 1")
        objects28[92] = point401
        point402 = workPart.Points.FindObject("ENTITY 2 208 1")
        objects28[93] = point402
        point403 = workPart.Points.FindObject("ENTITY 2 207 1")
        objects28[94] = point403
        point404 = workPart.Points.FindObject("ENTITY 2 206 1")
        objects28[95] = point404
        point405 = workPart.Points.FindObject("ENTITY 2 205 1")
        objects28[96] = point405
        point406 = workPart.Points.FindObject("ENTITY 2 204 1")
        objects28[97] = point406
        point407 = workPart.Points.FindObject("ENTITY 2 203 1")
        objects28[98] = point407
        point408 = workPart.Points.FindObject("ENTITY 2 202 1")
        objects28[99] = point408
        point409 = workPart.Points.FindObject("ENTITY 2 201 1")
        objects28[100] = point409
        point410 = workPart.Points.FindObject("ENTITY 2 200 1")
        objects28[101] = point410
        point411 = workPart.Points.FindObject("ENTITY 2 199 1")
        objects28[102] = point411
        point412 = workPart.Points.FindObject("ENTITY 2 198 1")
        objects28[103] = point412
        point413 = workPart.Points.FindObject("ENTITY 2 197 1")
        objects28[104] = point413
        point414 = workPart.Points.FindObject("ENTITY 2 196 1")
        objects28[105] = point414
        point415 = workPart.Points.FindObject("ENTITY 2 195 1")
        objects28[106] = point415
        point416 = workPart.Points.FindObject("ENTITY 2 194 1")
        objects28[107] = point416
        point417 = workPart.Points.FindObject("ENTITY 2 193 1")
        objects28[108] = point417
        point418 = workPart.Points.FindObject("ENTITY 2 192 1")
        objects28[109] = point418
        point419 = workPart.Points.FindObject("ENTITY 2 191 1")
        objects28[110] = point419
        point420 = workPart.Points.FindObject("ENTITY 2 190 1")
        objects28[111] = point420
        point421 = workPart.Points.FindObject("ENTITY 2 189 1")
        objects28[112] = point421
        point422 = workPart.Points.FindObject("ENTITY 2 188 1")
        objects28[113] = point422
        point423 = workPart.Points.FindObject("ENTITY 2 187 1")
        objects28[114] = point423
        point424 = workPart.Points.FindObject("ENTITY 2 186 1")
        objects28[115] = point424
        point425 = workPart.Points.FindObject("ENTITY 2 185 1")
        objects28[116] = point425
        point426 = workPart.Points.FindObject("ENTITY 2 184 1")
        objects28[117] = point426
        point427 = workPart.Points.FindObject("ENTITY 2 183 1")
        objects28[118] = point427
        point428 = workPart.Points.FindObject("ENTITY 2 182 1")
        objects28[119] = point428
        point429 = workPart.Points.FindObject("ENTITY 2 181 1")
        objects28[120] = point429
        point430 = workPart.Points.FindObject("ENTITY 2 180 1")
        objects28[121] = point430
        point431 = workPart.Points.FindObject("ENTITY 2 179 1")
        objects28[122] = point431
        point432 = workPart.Points.FindObject("ENTITY 2 178 1")
        objects28[123] = point432
        point433 = workPart.Points.FindObject("ENTITY 2 177 1")
        objects28[124] = point433
        point434 = workPart.Points.FindObject("ENTITY 2 176 1")
        objects28[125] = point434
        point435 = workPart.Points.FindObject("ENTITY 2 175 1")
        objects28[126] = point435
        point436 = workPart.Points.FindObject("ENTITY 2 174 1")
        objects28[127] = point436
        point437 = workPart.Points.FindObject("ENTITY 2 173 1")
        objects28[128] = point437
        point438 = workPart.Points.FindObject("ENTITY 2 172 1")
        objects28[129] = point438
        point439 = workPart.Points.FindObject("ENTITY 2 171 1")
        objects28[130] = point439
        point440 = workPart.Points.FindObject("ENTITY 2 170 1")
        objects28[131] = point440
        point441 = workPart.Points.FindObject("ENTITY 2 169 1")
        objects28[132] = point441
        point442 = workPart.Points.FindObject("ENTITY 2 168 1")
        objects28[133] = point442
        point443 = workPart.Points.FindObject("ENTITY 2 167 1")
        objects28[134] = point443
        point444 = workPart.Points.FindObject("ENTITY 2 166 1")
        objects28[135] = point444
        point445 = workPart.Points.FindObject("ENTITY 2 165 1")
        objects28[136] = point445
        point446 = workPart.Points.FindObject("ENTITY 2 164 1")
        objects28[137] = point446
        point447 = workPart.Points.FindObject("ENTITY 2 163 1")
        objects28[138] = point447
        point448 = workPart.Points.FindObject("ENTITY 2 162 1")
        objects28[139] = point448
        point449 = workPart.Points.FindObject("ENTITY 2 161 1")
        objects28[140] = point449
        point450 = workPart.Points.FindObject("ENTITY 2 160 1")
        objects28[141] = point450
        point451 = workPart.Points.FindObject("ENTITY 2 159 1")
        objects28[142] = point451
        point452 = workPart.Points.FindObject("ENTITY 2 158 1")
        objects28[143] = point452
        point453 = workPart.Points.FindObject("ENTITY 2 157 1")
        objects28[144] = point453
        point454 = workPart.Points.FindObject("ENTITY 2 156 1")
        objects28[145] = point454
        point455 = workPart.Points.FindObject("ENTITY 2 155 1")
        objects28[146] = point455
        point456 = workPart.Points.FindObject("ENTITY 2 154 1")
        objects28[147] = point456
        point457 = workPart.Points.FindObject("ENTITY 2 153 1")
        objects28[148] = point457
        point458 = workPart.Points.FindObject("ENTITY 2 152 1")
        objects28[149] = point458
        point459 = workPart.Points.FindObject("ENTITY 2 151 1")
        objects28[150] = point459
        point460 = workPart.Points.FindObject("ENTITY 2 150 1")
        objects28[151] = point460
        point461 = workPart.Points.FindObject("ENTITY 2 149 1")
        objects28[152] = point461
        point462 = workPart.Points.FindObject("ENTITY 2 148 1")
        objects28[153] = point462
        point463 = workPart.Points.FindObject("ENTITY 2 147 1")
        objects28[154] = point463
        point464 = workPart.Points.FindObject("ENTITY 2 146 1")
        objects28[155] = point464
        point465 = workPart.Points.FindObject("ENTITY 2 145 1")
        objects28[156] = point465
        point466 = workPart.Points.FindObject("ENTITY 2 144 1")
        objects28[157] = point466
        point467 = workPart.Points.FindObject("ENTITY 2 143 1")
        objects28[158] = point467
        point468 = workPart.Points.FindObject("ENTITY 2 142 1")
        objects28[159] = point468
        point469 = workPart.Points.FindObject("ENTITY 2 141 1")
        objects28[160] = point469
        point470 = workPart.Points.FindObject("ENTITY 2 140 1")
        objects28[161] = point470
        point471 = workPart.Points.FindObject("ENTITY 2 139 1")
        objects28[162] = point471
        point472 = workPart.Points.FindObject("ENTITY 2 138 1")
        objects28[163] = point472
        point473 = workPart.Points.FindObject("ENTITY 2 137 1")
        objects28[164] = point473
        point474 = workPart.Points.FindObject("ENTITY 2 136 1")
        objects28[165] = point474
        point475 = workPart.Points.FindObject("ENTITY 2 135 1")
        objects28[166] = point475
        point476 = workPart.Points.FindObject("ENTITY 2 134 1")
        objects28[167] = point476
        point477 = workPart.Points.FindObject("ENTITY 2 133 1")
        objects28[168] = point477
        point478 = workPart.Points.FindObject("ENTITY 2 132 1")
        objects28[169] = point478
        point479 = workPart.Points.FindObject("ENTITY 2 131 1")
        objects28[170] = point479
        point480 = workPart.Points.FindObject("ENTITY 2 130 1")
        objects28[171] = point480
        point481 = workPart.Points.FindObject("ENTITY 2 129 1")
        objects28[172] = point481
        point482 = workPart.Points.FindObject("ENTITY 2 128 1")
        objects28[173] = point482
        point483 = workPart.Points.FindObject("ENTITY 2 127 1")
        objects28[174] = point483
        point484 = workPart.Points.FindObject("ENTITY 2 126 1")
        objects28[175] = point484
        point485 = workPart.Points.FindObject("ENTITY 2 125 1")
        objects28[176] = point485
        point486 = workPart.Points.FindObject("ENTITY 2 124 1")
        objects28[177] = point486
        point487 = workPart.Points.FindObject("ENTITY 2 123 1")
        objects28[178] = point487
        point488 = workPart.Points.FindObject("ENTITY 2 122 1")
        objects28[179] = point488
        point489 = workPart.Points.FindObject("ENTITY 2 121 1")
        objects28[180] = point489
        point490 = workPart.Points.FindObject("ENTITY 2 120 1")
        objects28[181] = point490
        point491 = workPart.Points.FindObject("ENTITY 2 119 1")
        objects28[182] = point491
        point492 = workPart.Points.FindObject("ENTITY 2 118 1")
        objects28[183] = point492
        point493 = workPart.Points.FindObject("ENTITY 2 117 1")
        objects28[184] = point493
        point494 = workPart.Points.FindObject("ENTITY 2 116 1")
        objects28[185] = point494
        point495 = workPart.Points.FindObject("ENTITY 2 115 1")
        objects28[186] = point495
        point496 = workPart.Points.FindObject("ENTITY 2 114 1")
        objects28[187] = point496
        point497 = workPart.Points.FindObject("ENTITY 2 113 1")
        objects28[188] = point497
        point498 = workPart.Points.FindObject("ENTITY 2 112 1")
        objects28[189] = point498
        point499 = workPart.Points.FindObject("ENTITY 2 111 1")
        objects28[190] = point499
        point500 = workPart.Points.FindObject("ENTITY 2 110 1")
        objects28[191] = point500
        point501 = workPart.Points.FindObject("ENTITY 2 109 1")
        objects28[192] = point501
        point502 = workPart.Points.FindObject("ENTITY 2 108 1")
        objects28[193] = point502
        point503 = workPart.Points.FindObject("ENTITY 2 107 1")
        objects28[194] = point503
        point504 = workPart.Points.FindObject("ENTITY 2 106 1")
        objects28[195] = point504
        point505 = workPart.Points.FindObject("ENTITY 2 105 1")
        objects28[196] = point505
        point506 = workPart.Points.FindObject("ENTITY 2 104 1")
        objects28[197] = point506
        point507 = workPart.Points.FindObject("ENTITY 2 103 1")
        objects28[198] = point507
        point508 = workPart.Points.FindObject("ENTITY 2 102 1")
        objects28[199] = point508
        point509 = workPart.Points.FindObject("ENTITY 2 101 1")
        objects28[200] = point509
        point510 = workPart.Points.FindObject("ENTITY 2 100 1")
        objects28[201] = point510
        point511 = workPart.Points.FindObject("ENTITY 2 99 1")
        objects28[202] = point511
        point512 = workPart.Points.FindObject("ENTITY 2 98 1")
        objects28[203] = point512
        point513 = workPart.Points.FindObject("ENTITY 2 97 1")
        objects28[204] = point513
        point514 = workPart.Points.FindObject("ENTITY 2 96 1")
        objects28[205] = point514
        point515 = workPart.Points.FindObject("ENTITY 2 95 1")
        objects28[206] = point515
        point516 = workPart.Points.FindObject("ENTITY 2 94 1")
        objects28[207] = point516
        point517 = workPart.Points.FindObject("ENTITY 2 93 1")
        objects28[208] = point517
        point518 = workPart.Points.FindObject("ENTITY 2 92 1")
        objects28[209] = point518
        point519 = workPart.Points.FindObject("ENTITY 2 91 1")
        objects28[210] = point519
        point520 = workPart.Points.FindObject("ENTITY 2 90 1")
        objects28[211] = point520
        point521 = workPart.Points.FindObject("ENTITY 2 89 1")
        objects28[212] = point521
        point522 = workPart.Points.FindObject("ENTITY 2 88 1")
        objects28[213] = point522
        point523 = workPart.Points.FindObject("ENTITY 2 87 1")
        objects28[214] = point523
        point524 = workPart.Points.FindObject("ENTITY 2 86 1")
        objects28[215] = point524
        point525 = workPart.Points.FindObject("ENTITY 2 85 1")
        objects28[216] = point525
        point526 = workPart.Points.FindObject("ENTITY 2 84 1")
        objects28[217] = point526
        point527 = workPart.Points.FindObject("ENTITY 2 83 1")
        objects28[218] = point527
        point528 = workPart.Points.FindObject("ENTITY 2 82 1")
        objects28[219] = point528
        point529 = workPart.Points.FindObject("ENTITY 2 81 1")
        objects28[220] = point529
        point530 = workPart.Points.FindObject("ENTITY 2 80 1")
        objects28[221] = point530
        point531 = workPart.Points.FindObject("ENTITY 2 79 1")
        objects28[222] = point531
        point532 = workPart.Points.FindObject("ENTITY 2 78 1")
        objects28[223] = point532
        point533 = workPart.Points.FindObject("ENTITY 2 77 1")
        objects28[224] = point533
        point534 = workPart.Points.FindObject("ENTITY 2 76 1")
        objects28[225] = point534
        point535 = workPart.Points.FindObject("ENTITY 2 75 1")
        objects28[226] = point535
        point536 = workPart.Points.FindObject("ENTITY 2 74 1")
        objects28[227] = point536
        point537 = workPart.Points.FindObject("ENTITY 2 73 1")
        objects28[228] = point537
        point538 = workPart.Points.FindObject("ENTITY 2 72 1")
        objects28[229] = point538
        point539 = workPart.Points.FindObject("ENTITY 2 71 1")
        objects28[230] = point539
        point540 = workPart.Points.FindObject("ENTITY 2 70 1")
        objects28[231] = point540
        point541 = workPart.Points.FindObject("ENTITY 2 69 1")
        objects28[232] = point541
        point542 = workPart.Points.FindObject("ENTITY 2 68 1")
        objects28[233] = point542
        point543 = workPart.Points.FindObject("ENTITY 2 67 1")
        objects28[234] = point543
        point544 = workPart.Points.FindObject("ENTITY 2 66 1")
        objects28[235] = point544
        point545 = workPart.Points.FindObject("ENTITY 2 65 1")
        objects28[236] = point545
        point546 = workPart.Points.FindObject("ENTITY 2 64 1")
        objects28[237] = point546
        point547 = workPart.Points.FindObject("ENTITY 2 63 1")
        objects28[238] = point547
        point548 = workPart.Points.FindObject("ENTITY 2 62 1")
        objects28[239] = point548
        point549 = workPart.Points.FindObject("ENTITY 2 61 1")
        objects28[240] = point549
        point550 = workPart.Points.FindObject("ENTITY 2 60 1")
        objects28[241] = point550
        point551 = workPart.Points.FindObject("ENTITY 2 59 1")
        objects28[242] = point551
        point552 = workPart.Points.FindObject("ENTITY 2 58 1")
        objects28[243] = point552
        point553 = workPart.Points.FindObject("ENTITY 2 57 1")
        objects28[244] = point553
        point554 = workPart.Points.FindObject("ENTITY 2 56 1")
        objects28[245] = point554
        point555 = workPart.Points.FindObject("ENTITY 2 55 1")
        objects28[246] = point555
        point556 = workPart.Points.FindObject("ENTITY 2 54 1")
        objects28[247] = point556
        point557 = workPart.Points.FindObject("ENTITY 2 53 1")
        objects28[248] = point557
        point558 = workPart.Points.FindObject("ENTITY 2 52 1")
        objects28[249] = point558
        point559 = workPart.Points.FindObject("ENTITY 2 51 1")
        objects28[250] = point559
        point560 = workPart.Points.FindObject("ENTITY 2 50 1")
        objects28[251] = point560
        point561 = workPart.Points.FindObject("ENTITY 2 49 1")
        objects28[252] = point561
        point562 = workPart.Points.FindObject("ENTITY 2 48 1")
        objects28[253] = point562
        point563 = workPart.Points.FindObject("ENTITY 2 47 1")
        objects28[254] = point563
        point564 = workPart.Points.FindObject("ENTITY 2 46 1")
        objects28[255] = point564
        point565 = workPart.Points.FindObject("ENTITY 2 45 1")
        objects28[256] = point565
        point566 = workPart.Points.FindObject("ENTITY 2 44 1")
        objects28[257] = point566
        point567 = workPart.Points.FindObject("ENTITY 2 43 1")
        objects28[258] = point567
        point568 = workPart.Points.FindObject("ENTITY 2 42 1")
        objects28[259] = point568
        point569 = workPart.Points.FindObject("ENTITY 2 41 1")
        objects28[260] = point569
        point570 = workPart.Points.FindObject("ENTITY 2 40 1")
        objects28[261] = point570
        point571 = workPart.Points.FindObject("ENTITY 2 39 1")
        objects28[262] = point571
        point572 = workPart.Points.FindObject("ENTITY 2 38 1")
        objects28[263] = point572
        point573 = workPart.Points.FindObject("ENTITY 2 37 1")
        objects28[264] = point573
        point574 = workPart.Points.FindObject("ENTITY 2 36 1")
        objects28[265] = point574
        point575 = workPart.Points.FindObject("ENTITY 2 35 1")
        objects28[266] = point575
        point576 = workPart.Points.FindObject("ENTITY 2 34 1")
        objects28[267] = point576
        point577 = workPart.Points.FindObject("ENTITY 2 33 1")
        objects28[268] = point577
        point578 = workPart.Points.FindObject("ENTITY 2 32 1")
        objects28[269] = point578
        point579 = workPart.Points.FindObject("ENTITY 2 31 1")
        objects28[270] = point579
        point580 = workPart.Points.FindObject("ENTITY 2 30 1")
        objects28[271] = point580
        point581 = workPart.Points.FindObject("ENTITY 2 29 1")
        objects28[272] = point581
        point582 = workPart.Points.FindObject("ENTITY 2 28 1")
        objects28[273] = point582
        point583 = workPart.Points.FindObject("ENTITY 2 27 1")
        objects28[274] = point583
        point584 = workPart.Points.FindObject("ENTITY 2 26 1")
        objects28[275] = point584
        point585 = workPart.Points.FindObject("ENTITY 2 25 1")
        objects28[276] = point585
        point586 = workPart.Points.FindObject("ENTITY 2 24 1")
        objects28[277] = point586
        point587 = workPart.Points.FindObject("ENTITY 2 23 1")
        objects28[278] = point587
        point588 = workPart.Points.FindObject("ENTITY 2 22 1")
        objects28[279] = point588
        point589 = workPart.Points.FindObject("ENTITY 2 21 1")
        objects28[280] = point589
        point590 = workPart.Points.FindObject("ENTITY 2 20 1")
        objects28[281] = point590
        point591 = workPart.Points.FindObject("ENTITY 2 19 1")
        objects28[282] = point591
        point592 = workPart.Points.FindObject("ENTITY 2 18 1")
        objects28[283] = point592
        point593 = workPart.Points.FindObject("ENTITY 2 17 1")
        objects28[284] = point593
        point594 = workPart.Points.FindObject("ENTITY 2 16 1")
        objects28[285] = point594
        point595 = workPart.Points.FindObject("ENTITY 2 15 1")
        objects28[286] = point595
        point596 = workPart.Points.FindObject("ENTITY 2 14 1")
        objects28[287] = point596
        point597 = workPart.Points.FindObject("ENTITY 2 13 1")
        objects28[288] = point597
        point598 = workPart.Points.FindObject("ENTITY 2 12 1")
        objects28[289] = point598
        point599 = workPart.Points.FindObject("ENTITY 2 11 1")
        objects28[290] = point599
        point600 = workPart.Points.FindObject("ENTITY 2 10 1")
        objects28[291] = point600
        point601 = workPart.Points.FindObject("ENTITY 2 9 1")
        objects28[292] = point601
        point602 = workPart.Points.FindObject("ENTITY 2 8 1")
        objects28[293] = point602
        point603 = workPart.Points.FindObject("ENTITY 2 7 1")
        objects28[294] = point603
        point604 = workPart.Points.FindObject("ENTITY 2 6 1")
        objects28[295] = point604
        point605 = workPart.Points.FindObject("ENTITY 2 5 1")
        objects28[296] = point605
        point606 = workPart.Points.FindObject("ENTITY 2 4 1")
        objects28[297] = point606
        point607 = workPart.Points.FindObject("ENTITY 2 3 1")
        objects28[298] = point607
        point608 = workPart.Points.FindObject("ENTITY 2 2 1")
        objects28[299] = point608
        point609 = workPart.Points.FindObject("ENTITY 2 1 1")
        objects28[300] = point609
        group2 = nXObject9
        objects28[301] = group2
        added3 = fitCurveBuilder2.Target.Add(objects28)
        
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
        helpPoint5 = NXOpen.Point3d(0.94511024875883365, 0.0019270956380859523, 0.00025981659128438426)
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
        
        objects29 = [NXOpen.DisplayableObject.Null] * 1 
        compositeCurve3 = nXObject11
        spline4 = compositeCurve3.FindObject("CURVE 1")
        objects29[0] = spline4
        theSession.DisplayManager.BlankObjects(objects29)
        
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
        
        objects30 = [NXOpen.TaggedObject.Null] * 303 
        objects30[0] = point344
        objects30[1] = point345
        objects30[2] = point346
        objects30[3] = point347
        objects30[4] = point348
        objects30[5] = point349
        objects30[6] = point350
        objects30[7] = point351
        objects30[8] = point352
        objects30[9] = point353
        objects30[10] = point354
        objects30[11] = point355
        objects30[12] = point498
        objects30[13] = point499
        objects30[14] = point500
        objects30[15] = point501
        objects30[16] = point502
        objects30[17] = point503
        objects30[18] = point504
        objects30[19] = point505
        objects30[20] = point506
        objects30[21] = point507
        objects30[22] = point508
        objects30[23] = point509
        objects30[24] = point510
        objects30[25] = point511
        objects30[26] = point391
        objects30[27] = point392
        objects30[28] = point393
        objects30[29] = point394
        objects30[30] = point395
        objects30[31] = point396
        objects30[32] = point397
        objects30[33] = point398
        objects30[34] = point399
        objects30[35] = point400
        objects30[36] = point401
        objects30[37] = point402
        objects30[38] = point403
        objects30[39] = point404
        objects30[40] = point405
        objects30[41] = point406
        objects30[42] = point407
        objects30[43] = point408
        objects30[44] = point409
        objects30[45] = point410
        objects30[46] = point411
        objects30[47] = point412
        objects30[48] = point413
        objects30[49] = point414
        objects30[50] = point415
        objects30[51] = point416
        objects30[52] = point417
        objects30[53] = point418
        objects30[54] = point419
        objects30[55] = point420
        objects30[56] = point421
        objects30[57] = point422
        objects30[58] = point423
        objects30[59] = point424
        objects30[60] = point462
        objects30[61] = point463
        objects30[62] = point464
        objects30[63] = point465
        objects30[64] = point466
        objects30[65] = point467
        objects30[66] = point468
        objects30[67] = point469
        objects30[68] = point470
        objects30[69] = point471
        objects30[70] = point472
        objects30[71] = point473
        objects30[72] = point474
        objects30[73] = point475
        objects30[74] = point476
        objects30[75] = point477
        objects30[76] = point478
        objects30[77] = point479
        objects30[78] = point480
        objects30[79] = point481
        objects30[80] = point482
        objects30[81] = point483
        objects30[82] = point484
        objects30[83] = point485
        objects30[84] = point486
        objects30[85] = point487
        objects30[86] = point488
        objects30[87] = point489
        objects30[88] = point490
        objects30[89] = point491
        objects30[90] = point492
        objects30[91] = point493
        objects30[92] = point494
        objects30[93] = point495
        objects30[94] = point496
        objects30[95] = point497
        objects30[96] = point340
        objects30[97] = point341
        objects30[98] = point342
        objects30[99] = point343
        objects30[100] = point595
        objects30[101] = point596
        objects30[102] = point597
        objects30[103] = point598
        objects30[104] = point599
        objects30[105] = point600
        objects30[106] = point601
        objects30[107] = point602
        objects30[108] = point603
        objects30[109] = point604
        objects30[110] = point605
        objects30[111] = point606
        objects30[112] = point607
        objects30[113] = point608
        objects30[114] = point609
        objects30[115] = group2
        objects30[116] = spline3
        objects30[117] = point364
        objects30[118] = point365
        objects30[119] = point366
        objects30[120] = point367
        objects30[121] = point368
        objects30[122] = point369
        objects30[123] = point370
        objects30[124] = point371
        objects30[125] = point372
        objects30[126] = point373
        objects30[127] = point374
        objects30[128] = point375
        objects30[129] = point376
        objects30[130] = point377
        objects30[131] = point378
        objects30[132] = point379
        objects30[133] = point380
        objects30[134] = point381
        objects30[135] = point382
        objects30[136] = point383
        objects30[137] = point384
        objects30[138] = point385
        objects30[139] = point386
        objects30[140] = point387
        objects30[141] = point388
        objects30[142] = point389
        objects30[143] = point390
        objects30[144] = point557
        objects30[145] = point558
        objects30[146] = point559
        objects30[147] = point560
        objects30[148] = point561
        objects30[149] = point562
        objects30[150] = point563
        objects30[151] = point564
        objects30[152] = point565
        objects30[153] = point566
        objects30[154] = point567
        objects30[155] = point568
        objects30[156] = point569
        objects30[157] = point570
        objects30[158] = point571
        objects30[159] = point572
        objects30[160] = point573
        objects30[161] = point574
        objects30[162] = point575
        objects30[163] = point576
        objects30[164] = point577
        objects30[165] = point578
        objects30[166] = point579
        objects30[167] = point580
        objects30[168] = point581
        objects30[169] = point582
        objects30[170] = point583
        objects30[171] = point584
        objects30[172] = point585
        objects30[173] = point586
        objects30[174] = point587
        objects30[175] = point588
        objects30[176] = point589
        objects30[177] = point590
        objects30[178] = point591
        objects30[179] = point592
        objects30[180] = point593
        objects30[181] = point594
        objects30[182] = point356
        objects30[183] = point357
        objects30[184] = point358
        objects30[185] = point359
        objects30[186] = point360
        objects30[187] = point361
        objects30[188] = point362
        objects30[189] = point363
        objects30[190] = point309
        objects30[191] = point310
        objects30[192] = point311
        objects30[193] = point312
        objects30[194] = point313
        objects30[195] = point314
        objects30[196] = point315
        objects30[197] = point316
        objects30[198] = point317
        objects30[199] = point318
        objects30[200] = point319
        objects30[201] = point320
        objects30[202] = point321
        objects30[203] = point322
        objects30[204] = point323
        objects30[205] = point324
        objects30[206] = point325
        objects30[207] = point326
        objects30[208] = point327
        objects30[209] = point328
        objects30[210] = point329
        objects30[211] = point330
        objects30[212] = point331
        objects30[213] = point332
        objects30[214] = point333
        objects30[215] = point334
        objects30[216] = point335
        objects30[217] = point336
        objects30[218] = point337
        objects30[219] = point338
        objects30[220] = point339
        objects30[221] = point512
        objects30[222] = point513
        objects30[223] = point514
        objects30[224] = point515
        objects30[225] = point516
        objects30[226] = point517
        objects30[227] = point518
        objects30[228] = point519
        objects30[229] = point520
        objects30[230] = point521
        objects30[231] = point522
        objects30[232] = point523
        objects30[233] = point524
        objects30[234] = point525
        objects30[235] = point526
        objects30[236] = point527
        objects30[237] = point528
        objects30[238] = point529
        objects30[239] = point530
        objects30[240] = point531
        objects30[241] = point532
        objects30[242] = point533
        objects30[243] = point534
        objects30[244] = point535
        objects30[245] = point536
        objects30[246] = point537
        objects30[247] = point538
        objects30[248] = point539
        objects30[249] = point540
        objects30[250] = point541
        objects30[251] = point542
        objects30[252] = point543
        objects30[253] = point544
        objects30[254] = point545
        objects30[255] = point546
        objects30[256] = point547
        objects30[257] = point548
        objects30[258] = point549
        objects30[259] = point550
        objects30[260] = point551
        objects30[261] = point552
        objects30[262] = point553
        objects30[263] = point554
        objects30[264] = point555
        objects30[265] = point556
        objects30[266] = point425
        objects30[267] = point426
        objects30[268] = point427
        objects30[269] = point428
        objects30[270] = point429
        objects30[271] = point430
        objects30[272] = point431
        objects30[273] = point432
        objects30[274] = point433
        objects30[275] = point434
        objects30[276] = point435
        objects30[277] = point436
        objects30[278] = point437
        objects30[279] = point438
        objects30[280] = point439
        objects30[281] = point440
        objects30[282] = point441
        objects30[283] = point442
        objects30[284] = point443
        objects30[285] = point444
        objects30[286] = point445
        objects30[287] = point446
        objects30[288] = point447
        objects30[289] = point448
        objects30[290] = point449
        objects30[291] = point450
        objects30[292] = point451
        objects30[293] = point452
        objects30[294] = point453
        objects30[295] = point454
        objects30[296] = point455
        objects30[297] = point456
        objects30[298] = point457
        objects30[299] = point458
        objects30[300] = point459
        objects30[301] = point460
        objects30[302] = point461
        nErrs8 = theSession.UpdateManager.AddObjectsToDeleteList(objects30)
        
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
        
        objects31 = [NXOpen.DisplayableObject.Null] * 1 
        swept2 = nXObject12
        face19 = swept2.FindObject("FACE 10015 {(262.1086446359832,2.652894767263,-0.1236402380488) SWEPT(5)}")
        objects31[0] = face19
        displayModification2.Apply(objects31)
        
        face19.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects32 = [NXOpen.DisplayableObject.Null] * 1 
        face20 = swept2.FindObject("FACE 10014 {(262.1681905735861,1.7283056363762,0.8570856681183) SWEPT(5)}")
        objects32[0] = face20
        displayModification2.Apply(objects32)
        
        face20.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects33 = [NXOpen.DisplayableObject.Null] * 1 
        face21 = swept2.FindObject("FACE 10013 {(262.2102127462241,1.0470006826553,1.1584621695912) SWEPT(5)}")
        objects33[0] = face21
        displayModification2.Apply(objects33)
        
        face21.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects34 = [NXOpen.DisplayableObject.Null] * 1 
        face22 = swept2.FindObject("FACE 10012 {(262.2097265708398,1.0748973829203,1.4263888053355) SWEPT(5)}")
        objects34[0] = face22
        displayModification2.Apply(objects34)
        
        face22.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects35 = [NXOpen.DisplayableObject.Null] * 1 
        face23 = swept2.FindObject("FACE 10011 {(261.7553852964263,8.7270678484925,2.0455552986704) SWEPT(5)}")
        objects35[0] = face23
        displayModification2.Apply(objects35)
        
        face23.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects36 = [NXOpen.DisplayableObject.Null] * 1 
        face24 = swept2.FindObject("FACE 10010 {(261.2958821351687,16.3747984492826,1.4325827218921) SWEPT(5)}")
        objects36[0] = face24
        displayModification2.Apply(objects36)
        
        face24.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects37 = [NXOpen.DisplayableObject.Null] * 1 
        face25 = swept2.FindObject("FACE 10009 {(261.2923000123251,16.4158654833668,1.1762222860232) SWEPT(5)}")
        objects37[0] = face25
        displayModification2.Apply(objects37)
        
        face25.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects38 = [NXOpen.DisplayableObject.Null] * 1 
        face26 = swept2.FindObject("FACE 10008 {(261.3239361263919,15.8621941183211,0.8504292445884) SWEPT(5)}")
        objects38[0] = face26
        displayModification2.Apply(objects38)
        
        face26.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects39 = [NXOpen.DisplayableObject.Null] * 1 
        face27 = swept2.FindObject("FACE 10007 {(261.3625858626773,15.1389171971563,-0.1830357114156) SWEPT(5)}")
        objects39[0] = face27
        displayModification2.Apply(objects39)
        
        face27.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects40 = [NXOpen.DisplayableObject.Null] * 1 
        face28 = swept2.FindObject("FACE 10006 {(261.2872090901781,16.398127140349,-0.2201452141863) SWEPT(5)}")
        objects40[0] = face28
        displayModification2.Apply(objects40)
        
        face28.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects41 = [NXOpen.DisplayableObject.Null] * 1 
        face29 = swept2.FindObject("FACE 10005 {(261.2263342256081,17.4800025163761,0.6303655624164) SWEPT(5)}")
        objects41[0] = face29
        displayModification2.Apply(objects41)
        
        face29.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects42 = [NXOpen.DisplayableObject.Null] * 1 
        face30 = swept2.FindObject("FACE 10004 {(261.2494552739408,17.214932634233,2.2850609169071) SWEPT(5)}")
        objects42[0] = face30
        displayModification2.Apply(objects42)
        
        face30.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects43 = [NXOpen.DisplayableObject.Null] * 1 
        face31 = swept2.FindObject("FACE 10003 {(261.7606509529494,8.7272367116535,3.2433349308956) SWEPT(5)}")
        objects43[0] = face31
        displayModification2.Apply(objects43)
        
        face31.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects44 = [NXOpen.DisplayableObject.Null] * 1 
        face32 = swept2.FindObject("FACE 10016 {(262.1880082955782,1.3136677697039,-0.2664735861349) SWEPT(5)}")
        objects44[0] = face32
        displayModification2.Apply(objects44)
        
        face32.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects45 = [NXOpen.DisplayableObject.Null] * 1 
        face33 = swept2.FindObject("FACE 2 {(522.3445569741205,34.0158989637805,-0.3457365371509) SWEPT(5)}")
        objects45[0] = face33
        displayModification2.Apply(objects45)
        
        face33.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects46 = [NXOpen.DisplayableObject.Null] * 1 
        face34 = swept2.FindObject("FACE 10002 {(262.264933734811,0.2117301597589,2.2549991698254) SWEPT(5)}")
        objects46[0] = face34
        displayModification2.Apply(objects46)
        
        face34.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects47 = [NXOpen.DisplayableObject.Null] * 1 
        face35 = swept2.FindObject("FACE 1 {(0,1.0818839027918,1.9571319661241) SWEPT(5)}")
        objects47[0] = face35
        displayModification2.Apply(objects47)
        
        face35.Color = 32767
        
        displayModification2.SetNewGrid(0, 0)
        
        displayModification2.PoleDisplayState = False
        
        displayModification2.KnotDisplayState = False
        
        objects48 = [NXOpen.DisplayableObject.Null] * 1 
        face36 = swept2.FindObject("FACE 10001 {(262.268071785919,0.0316690665659,0.5256490148081) SWEPT(5)}")
        objects48[0] = face36
        displayModification2.Apply(objects48)
        
        face36.Color = 32767
        
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
        
        objects49 = [NXOpen.DisplayableObject.Null] * 1 
        body2 = workPart.Bodies.FindObject("SWEPT(5)")
        objects49[0] = body2
        theSession.DisplayManager.BlankObjects(objects49)
        
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
        
        booleanBuilder1.CopyTargets = True
        
        booleanBuilder1.CopyTools = True
        
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
        
        markId83 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "合并")
        
        theSession.DeleteUndoMark(markId83, None)
        
        markId84 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "合并")
        
        nXObject13 = booleanBuilder1.Commit()
        
        theSession.DeleteUndoMark(markId84, None)
        
        theSession.SetUndoMarkName(markId82, "合并")
        
        booleanBuilder1.Destroy()
        
        # ----------------------------------------------
        #   菜单：分析(L)  ->测量(S)...
        # ----------------------------------------------
        markId85 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "起点")
        
        theSession.SetUndoMarkName(markId85, "测量 对话框")
        
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
        
        markId86 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        markId87 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Measurement Apply")
        
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
        
        markId88 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Measurement Update")
        
        nErrs10 = theSession.UpdateManager.DoUpdate(markId88)
        
        theSession.DeleteUndoMark(markId88, "Measurement Update")
        
        theSession.DeleteUndoMark(markId87, "Measurement Apply")
        
        datadeleted1 = theSession.DeleteTransientDynamicSectionCutData()
        
        theSession.DeleteUndoMark(markId86, None)
        
        theSession.SetUndoMarkName(markId85, "测量")
        
        scCollector7.Destroy()
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        markId89 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Start")
        
        theSession.SetUndoMarkName(markId89, "测量 对话框")
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector8 = workPart.ScCollectors.CreateCollector()
        
        scCollector8.SetMultiComponent()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        # ----------------------------------------------
        #   对话开始 测量
        # ----------------------------------------------
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
        
        markId90 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        markId91 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Measurement Apply")
        
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
        
        markId92 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Measurement Update")
        
        nErrs11 = theSession.UpdateManager.DoUpdate(markId92)
        
        theSession.DeleteUndoMark(markId92, "Measurement Update")
        
        theSession.DeleteUndoMark(markId91, "Measurement Apply")
        
        datadeleted2 = theSession.DeleteTransientDynamicSectionCutData()
        
        theSession.DeleteUndoMark(markId90, None)
        
        theSession.SetUndoMarkName(markId89, "测量")
        
        scCollector9.Destroy()
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        markId93 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Start")
        
        theSession.SetUndoMarkName(markId93, "测量 对话框")
        
        workPart.MeasureManager.SetPartTransientModification()
        
        scCollector10 = workPart.ScCollectors.CreateCollector()
        
        scCollector10.SetMultiComponent()
        
        workPart.MeasureManager.SetPartTransientModification()
        
        # ----------------------------------------------
        #   对话开始 测量
        # ----------------------------------------------
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
        
        markId94 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        theSession.DeleteUndoMark(markId94, None)
        
        markId95 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "测量")
        
        markId96 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Measurement Apply")
        
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
        
        markId97 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Measurement Update")
        
        nErrs12 = theSession.UpdateManager.DoUpdate(markId97)
        
        theSession.DeleteUndoMark(markId97, "Measurement Update")
        
        theSession.DeleteUndoMark(markId96, "Measurement Apply")
        
        datadeleted3 = theSession.DeleteTransientDynamicSectionCutData()
        
        theSession.DeleteUndoMark(markId95, None)
        
        theSession.SetUndoMarkName(markId93, "测量")
        
        scCollector11.Destroy()
        
        workPart.MeasureManager.ClearPartTransientModification()
        
        markId98 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects50 = [NXOpen.DisplayableObject.Null] * 1 
        body3 = workPart.Bodies.FindObject("UNITE(6)")
        objects50[0] = body3
        theSession.DisplayManager.BlankObjects(objects50)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        markId99 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects51 = [NXOpen.DisplayableObject.Null] * 1 
        generalLabel1 = annotation1
        objects51[0] = generalLabel1
        theSession.DisplayManager.BlankObjects(objects51)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        markId100 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects52 = [NXOpen.DisplayableObject.Null] * 1 
        generalLabel2 = annotation2
        objects52[0] = generalLabel2
        theSession.DisplayManager.BlankObjects(objects52)
        
        workPart.ModelingViews.WorkView.FitAfterShowOrHide(NXOpen.View.ShowOrHideType.HideOnly)
        
        markId101 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Hide")
        
        objects53 = [NXOpen.DisplayableObject.Null] * 1 
        generalLabel3 = annotation3
        objects53[0] = generalLabel3
        theSession.DisplayManager.BlankObjects(objects53)
        
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
        #   菜单：工具(T)->重复命令(R)->4 停止操作记录录制
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
        strip_section_stp_path = f"{data_root}\\strip_section_stp\\type_2\\strip_section_{i:0>4}.stp"
        iou_3d_prt_path = f"{data_root}\\iou_3d_prt\\type_2\\iou_3d_{i:0>4}.prt"
        iou_3d_txt_path = f"{data_root}\\iou_3d_txt\\type_2\\iou_3d_{i:0>4}.txt"
        springback_strip_line_path = f"{data_root}\\springback_strip_line\\type_2\\springback_strip_line_{i:0>4}.txt"
        prediction_line_path = f"{data_root}\\prediction_line\\type_2\\{i:0>4}.txt"
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
    sys.stdout = open(f"{data_root}\\iou_3d_txt\\type_2\\iou_3d_total.txt", 'w', encoding = "utf-8")
    print(iou_3d / cnt)
    print(cnt)
    print(error)
    sys.stdout = sys.__stdout__