def DefaultValuepresent : I32EnumAttrCase<"present", 0>;
def DefaultValuenone : I32EnumAttrCase<"none", 1>;
def DefaultValue: I32EnumAttr<
  "ClauseDefaultValue",
  "DefaultValue Clause",
  [DefaultValuepresent,DefaultValuenone]> {
    let cppNamespace = "::mlir::acc";
    let genSpecializedAttr = 0;
}
def DefaultValueAttr : EnumAttr<OpenACC_Dialect, DefaultValue, "defaultvalue">;
