def DefaultValuepresent : I32EnumAttrCase<"Present", 0, "present">;
def DefaultValuenone : I32EnumAttrCase<"None", 1, "none">;
def DefaultValue: I32EnumAttr<
  "ClauseDefaultValue",
  "DefaultValue Clause",
  [DefaultValuepresent,DefaultValuenone]> {
    let cppNamespace = "::mlir::acc";
    let genSpecializedAttr = 0;
}
def DefaultValueAttr : EnumAttr<OpenACC_Dialect, DefaultValue, "defaultvalue">;
