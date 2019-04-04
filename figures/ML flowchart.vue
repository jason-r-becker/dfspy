<!-- Tufts VUE 3.3.0 concept-map (ML flowchart.vue) 2019-04-04 -->
<!-- Tufts VUE: http://vue.tufts.edu/ -->
<!-- Do Not Remove: VUE mapping @version(1.1) jar:file:/Applications/VUE.app/Contents/Resources/Java/VUE.jar!/tufts/vue/resources/lw_mapping_1_1.xml -->
<!-- Do Not Remove: Saved date Thu Apr 04 10:41:15 CDT 2019 by jasonbecker on platform Mac OS X 10.14.4 in JVM 1.8.0_121-b13 -->
<!-- Do Not Remove: Saving version @(#)VUE: built October 8 2015 at 1658 by tomadm on Linux 2.6.32-504.23.4.el6.x86_64 i386 JVM 1.7.0_21-b11(bits=32) -->
<?xml version="1.0" encoding="US-ASCII"?>
<LW-MAP xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="none" ID="0" label="ML flowchart.vue"
    created="1554310875701" x="0.0" y="0.0" width="1.4E-45"
    height="1.4E-45" strokeWidth="0.0" autoSized="false">
    <resource referenceCreated="1554392475482" size="26427"
        spec="/Users/jasonbecker/PycharmProjects/dfspy/figures/ML flowchart.vue"
        type="1" xsi:type="URLResource">
        <title>ML flowchart.vue</title>
        <property key="File" value="/Users/jasonbecker/PycharmProjects/dfspy/figures/ML flowchart.vue"/>
    </resource>
    <fillColor>#FFFFFF</fillColor>
    <strokeColor>#404040</strokeColor>
    <textColor>#000000</textColor>
    <font>SansSerif-plain-14</font>
    <URIString>http://vue.tufts.edu/rdf/resource/e438e85ec0a8016d2f1d9753c9ba885c</URIString>
    <child ID="6" label="Normalize Values&#xa;and Store Transformation"
        layerID="1" created="1554310957705" x="170.0" y="66.0"
        width="149.0" height="56.0" strokeWidth="1.0" autoSized="false" xsi:type="node">
        <fillColor>#FFFFFF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e85ec0a8016d2f1d975379a899aa</URIString>
        <shape xsi:type="rectangle"/>
    </child>
    <child ID="7" label="For each stat&#xa;in Essential Stats"
        layerID="1" created="1554311015732" x="-62.0" y="232.0"
        width="142.0" height="80.3" strokeWidth="1.0" autoSized="false" xsi:type="node">
        <fillColor>#FFFFFF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e85fc0a8016d2f1d9753de2247ab</URIString>
        <shape xsi:type="hexagon"/>
    </child>
    <child ID="8" label="Quantize Values&#xa;into 5 Quintiles"
        layerID="1" created="1554311045414" x="355.0" y="67.0"
        width="135.0" height="55.0" strokeWidth="1.0" autoSized="false" xsi:type="node">
        <fillColor>#FFFFFF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e85fc0a8016d2f1d97533b4399ba</URIString>
        <shape xsi:type="rectangle"/>
    </child>
    <child ID="9"
        label="Split into:&#xa;Train: Weeks 1-11&#xa;Cross-Validation: Weeks 12-14&#xa;Test: Weeks 15-17"
        layerID="1" created="1554311082842" x="106.0" y="195.0"
        width="277.2" height="156.20001" strokeWidth="1.0"
        autoSized="true" xsi:type="node">
        <fillColor>#FFFFFF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e860c0a8016d2f1d9753df417f6f</URIString>
        <shape xsi:type="diamond"/>
    </child>
    <child ID="10"
        label="Create 5 K-folds&#xa;with Same Distribution&#xa;of Quntiles in Each"
        layerID="1" created="1554311205436" x="527.0" y="68.0"
        width="158.0" height="55.0" strokeWidth="1.0" autoSized="false" xsi:type="node">
        <fillColor>#FFFFFF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e861c0a8016d2f1d975334f0c582</URIString>
        <shape xsi:type="rectangle"/>
    </child>
    <child ID="11"
        label="Using &quot;Leave 1 Out&quot;&#xa;K-fold Validation,&#xa;Identify and Store&#xa;Optimal Parameters"
        layerID="1" created="1554311274163" x="888.0" y="64.0"
        width="161.0" height="64.0" strokeWidth="1.0" autoSized="false" xsi:type="node">
        <fillColor>#FFFFFF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e861c0a8016d2f1d9753754bfe8c</URIString>
        <shape xsi:type="rectangle"/>
    </child>
    <child ID="13" label="Saved Scaler&#xa;(stat)" layerID="1"
        created="1554311387083" x="194.0" y="-17.5" width="101.299995"
        height="48.299995" strokeWidth="1.0" autoSized="true" xsi:type="node">
        <fillColor>#F4E5FF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e861c0a8016d2f1d97535afa4c57</URIString>
        <shape xsi:type="ellipse"/>
    </child>
    <child ID="14" label="Saved Model&#xa;(stat, ML algorithm)"
        layerID="1" created="1554311420425" x="899.35" y="-20.649998"
        width="134.59999" height="50.600002" strokeWidth="1.0"
        autoSized="true" xsi:type="node">
        <fillColor>#C6E8FF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e862c0a8016d2f1d9753826febac</URIString>
        <shape xsi:type="ellipse"/>
    </child>
    <child ID="15" label="Saved Scaler&#xa;(stat)" layerID="1"
        created="1554311469735" x="343.35" y="353.35" width="101.299995"
        height="48.299995" strokeWidth="1.0" autoSized="true" xsi:type="node">
        <fillColor>#F4E5FF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e862c0a8016d2f1d97535c554006</URIString>
        <shape xsi:type="ellipse"/>
    </child>
    <child ID="16" label="Normalize Values&#xa;Using Saved Scaler"
        layerID="1" created="1554311478427" x="462.5" y="244.5"
        width="149.0" height="56.0" strokeWidth="1.0" autoSized="false" xsi:type="node">
        <fillColor>#FFFFFF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e863c0a8016d2f1d975323dc6120</URIString>
        <shape xsi:type="rectangle"/>
    </child>
    <child ID="17" label="Normalize Values&#xa;Using Saved Scaler"
        layerID="1" created="1554311502784" x="169.5" y="455.5"
        width="149.0" height="56.0" strokeWidth="1.0" autoSized="false" xsi:type="node">
        <fillColor>#FFFFFF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e863c0a8016d2f1d97537aeab572</URIString>
        <shape xsi:type="rectangle"/>
    </child>
    <child ID="19" label="For each Algorithm&#xa;in ML Algorithms"
        layerID="1" created="1554311520198" x="654.5" y="242.5"
        width="142.3" height="62.0" strokeWidth="1.0" autoSized="false" xsi:type="node">
        <fillColor>#FFFFFF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e863c0a8016d2f1d97532ac5e045</URIString>
        <shape xsi:type="hexagon"/>
    </child>
    <child ID="21" label="Project Values using&#xa;Saved ML Algorithm "
        layerID="1" created="1554311526937" x="851.5" y="244.5"
        width="149.0" height="56.0" strokeWidth="1.0" autoSized="false" xsi:type="node">
        <fillColor>#FFFFFF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e864c0a8016d2f1d9753da68511d</URIString>
        <shape xsi:type="rectangle"/>
    </child>
    <child ID="22" label="For each algorithm&#xa;in ML Algorithms"
        layerID="1" created="1554311601159" x="718.85" y="64.5"
        width="142.3" height="62.0" strokeWidth="1.0" autoSized="false" xsi:type="node">
        <fillColor>#FFFFFF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e864c0a8016d2f1d9753159ce22c</URIString>
        <shape xsi:type="hexagon"/>
    </child>
    <child ID="23"
        label="Project Values using&#xa;Optimal Saved ML Algorithm "
        layerID="1" created="1554311642984" x="504.5" y="456.5"
        width="169.0" height="56.0" strokeWidth="1.0" autoSized="false" xsi:type="node">
        <fillColor>#FFFFFF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e865c0a8016d2f1d97537f9b7e3b</URIString>
        <shape xsi:type="rectangle"/>
    </child>
    <child ID="24" label="Saved Model&#xa;(stat, ML algorithm)"
        layerID="1" created="1554311652693" x="521.69995" y="351.2"
        width="134.59999" height="50.600002" strokeWidth="1.0"
        autoSized="true" xsi:type="node">
        <fillColor>#C6E8FF</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e865c0a8016d2f1d975316ef1d4d</URIString>
        <shape xsi:type="ellipse"/>
    </child>
    <child ID="41" layerID="1" created="1554311731390" x="79.2722"
        y="271.93536" width="28.212547" height="1.1097412"
        strokeWidth="1.0" autoSized="false" controlCount="0"
        arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e865c0a8016d2f1d9753e660d95f</URIString>
        <point1 x="79.77221" y="272.43536"/>
        <point2 x="106.98476" y="272.5451"/>
        <ID1 xsi:type="node">7</ID1>
        <ID2 xsi:type="node">9</ID2>
    </child>
    <child ID="42" label="Train" layerID="1" created="1554311732621"
        x="232.03601" y="121.5" width="25.0" height="74.03125"
        strokeWidth="1.0" autoSized="false" controlCount="0"
        arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e866c0a8016d2f1d97538c673b30</URIString>
        <point1 x="244.55641" y="195.03125"/>
        <point2 x="244.51562" y="122.0"/>
        <ID1 xsi:type="node">9</ID1>
        <ID2 xsi:type="node">6</ID2>
    </child>
    <child ID="43" label="CV" layerID="1" created="1554311734874"
        x="382.19714" y="266.23474" width="80.80289" height="13.0"
        strokeWidth="1.0" autoSized="false" controlCount="0"
        arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e866c0a8016d2f1d9753f7d8d7fc</URIString>
        <point1 x="382.6971" y="272.81662"/>
        <point2 x="462.5" y="272.65286"/>
        <ID1 xsi:type="node">9</ID1>
        <ID2 xsi:type="node">16</ID2>
    </child>
    <child ID="44" label="Test" layerID="1" created="1554311736560"
        x="233.22876" y="350.57812" width="22.0" height="105.421875"
        strokeWidth="1.0" autoSized="false" controlCount="0"
        arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e867c0a8016d2f1d975316153fd6</URIString>
        <point1 x="244.37766" y="351.07812"/>
        <point2 x="244.07985" y="455.5"/>
        <ID1 xsi:type="node">9</ID1>
        <ID2 xsi:type="node">17</ID2>
    </child>
    <child ID="45" layerID="1" created="1554311739400" x="611.0"
        y="272.39493" width="44.38861" height="1.2299805"
        strokeWidth="1.0" autoSized="false" controlCount="0"
        arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e868c0a8016d2f1d9753439edd6e</URIString>
        <point1 x="611.5" y="272.89493"/>
        <point2 x="654.8886" y="273.1249"/>
        <ID1 xsi:type="node">16</ID1>
        <ID2 xsi:type="node">19</ID2>
    </child>
    <child ID="46" layerID="1" created="1554311740222" x="795.93384"
        y="272.37186" width="56.0661" height="1.2748413"
        strokeWidth="1.0" autoSized="false" controlCount="0"
        arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e86ac0a8016d2f1d97534eaa38ca</URIString>
        <point1 x="796.4339" y="273.1467"/>
        <point2 x="851.5" y="272.87186"/>
        <ID1 xsi:type="node">19</ID1>
        <ID2 xsi:type="node">21</ID2>
    </child>
    <child ID="47" layerID="1" created="1554311741275" x="318.0"
        y="483.21594" width="187.0" height="1.5391235" strokeWidth="1.0"
        autoSized="false" controlCount="0" arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e86bc0a8016d2f1d9753f484ee09</URIString>
        <point1 x="318.5" y="483.71594"/>
        <point2 x="504.5" y="484.25507"/>
        <ID1 xsi:type="node">17</ID1>
        <ID2 xsi:type="node">23</ID2>
    </child>
    <child ID="48" layerID="1" created="1554311742461" x="318.5"
        y="93.70927" width="37.0" height="1.10112" strokeWidth="1.0"
        autoSized="false" controlCount="0" arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e86bc0a8016d2f1d975351f1647d</URIString>
        <point1 x="319.0" y="94.20927"/>
        <point2 x="355.0" y="94.31039"/>
        <ID1 xsi:type="node">6</ID1>
        <ID2 xsi:type="node">8</ID2>
    </child>
    <child ID="49" layerID="1" created="1554311743333" x="489.5"
        y="94.36786" width="38.0" height="1.2016373" strokeWidth="1.0"
        autoSized="false" controlCount="0" arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e86cc0a8016d2f1d9753da3de0c9</URIString>
        <point1 x="490.0" y="94.86785"/>
        <point2 x="527.0" y="95.06949"/>
        <ID1 xsi:type="node">8</ID1>
        <ID2 xsi:type="node">10</ID2>
    </child>
    <child ID="50" layerID="1" created="1554311744113" x="684.5"
        y="95.0" width="34.849976" height="1.0" strokeWidth="1.0"
        autoSized="false" controlCount="0" arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e86cc0a8016d2f1d97535e0dd271</URIString>
        <point1 x="685.0" y="95.5"/>
        <point2 x="718.85" y="95.5"/>
        <ID1 xsi:type="node">10</ID1>
        <ID2 xsi:type="node">22</ID2>
    </child>
    <child ID="51" layerID="1" created="1554311744870" x="860.4441"
        y="95.19872" width="28.05597" height="1.0757904"
        strokeWidth="1.0" autoSized="false" controlCount="0"
        arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e86dc0a8016d2f1d9753fd96189f</URIString>
        <point1 x="860.94403" y="95.69872"/>
        <point2 x="888.0" y="95.77451"/>
        <ID1 xsi:type="node">22</ID1>
        <ID2 xsi:type="node">11</ID2>
    </child>
    <child ID="52" layerID="1" created="1554311755697" x="421.04547"
        y="300.0" width="78.3212" height="57.774292" strokeWidth="1.0"
        strokeStyle="4" autoSized="false" controlCount="0"
        arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e86dc0a8016d2f1d97538b021701</URIString>
        <point1 x="421.54544" y="357.2743"/>
        <point2 x="498.86664" y="300.5"/>
        <ID1 xsi:type="node">15</ID1>
        <ID2 xsi:type="node">16</ID2>
    </child>
    <child ID="53" layerID="1" created="1554311757837" x="283.12265"
        y="397.02075" width="83.04608" height="58.979248"
        strokeWidth="1.0" strokeStyle="4" autoSized="false"
        controlCount="0" arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e86ec0a8016d2f1d9753f2af7a13</URIString>
        <point1 x="365.66873" y="397.52075"/>
        <point2 x="283.62265" y="455.5"/>
        <ID1 xsi:type="node">15</ID1>
        <ID2 xsi:type="node">17</ID2>
    </child>
    <child ID="54" layerID="1" created="1554311759505" x="640.2804"
        y="294.9911" width="211.7196" height="66.029175"
        strokeWidth="1.0" strokeStyle="4" autoSized="false"
        controlCount="0" arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e86ec0a8016d2f1d9753700b1698</URIString>
        <point1 x="640.7804" y="360.52026"/>
        <point2 x="851.5" y="295.4911"/>
        <ID1 xsi:type="node">24</ID1>
        <ID2 xsi:type="node">21</ID2>
    </child>
    <child ID="55" layerID="1" created="1554311760766" x="588.49994"
        y="401.30005" width="1.0" height="55.69998" strokeWidth="1.0"
        strokeStyle="4" autoSized="false" controlCount="0"
        arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e86fc0a8016d2f1d97539d8f3a4a</URIString>
        <point1 x="588.99994" y="401.80002"/>
        <point2 x="588.99994" y="456.5"/>
        <ID1 xsi:type="node">24</ID1>
        <ID2 xsi:type="node">23</ID2>
    </child>
    <child ID="56" layerID="1" created="1554311762964" x="966.662"
        y="29.433594" width="1.6899414" height="35.066406"
        strokeWidth="1.0" strokeStyle="2" autoSized="false"
        controlCount="0" arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e86fc0a8016d2f1d97531f88fd63</URIString>
        <point1 x="967.8519" y="64.0"/>
        <point2 x="967.162" y="29.933594"/>
        <ID1 xsi:type="node">11</ID1>
        <ID2 xsi:type="node">14</ID2>
    </child>
    <child ID="57" layerID="1" created="1554311764399" x="244.04807"
        y="30.3125" width="1.0604095" height="36.1875" strokeWidth="1.0"
        strokeStyle="2" autoSized="false" controlCount="0"
        arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e870c0a8016d2f1d975365d68971</URIString>
        <point1 x="244.54808" y="66.0"/>
        <point2 x="244.60849" y="30.8125"/>
        <ID1 xsi:type="node">6</ID1>
        <ID2 xsi:type="node">13</ID2>
    </child>
    <child ID="58" label="Save Stat Projections" layerID="1"
        created="1554311929679" x="1121.0" y="235.5" width="140.79999"
        height="75.0" strokeWidth="1.0" autoSized="false" xsi:type="node">
        <fillColor>#ECFFD4</fillColor>
        <strokeColor>#000000</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e870c0a8016d2f1d9753f1615c5f</URIString>
        <shape xsi:type="ellipse"/>
    </child>
    <child ID="59" layerID="1" created="1554312023994" x="668.25024"
        y="293.11993" width="464.91956" height="163.88007"
        strokeWidth="1.0" strokeStyle="2" autoSized="false"
        controlCount="0" arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e871c0a8016d2f1d9753a1d4dedf</URIString>
        <point1 x="668.75024" y="456.5"/>
        <point2 x="1132.6698" y="293.61993"/>
        <ID1 xsi:type="node">23</ID1>
        <ID2 xsi:type="node">58</ID2>
    </child>
    <child ID="61" layerID="1" created="1554312027847" x="1008.29834"
        y="127.5" width="144.38849" height="114.86163" strokeWidth="1.0"
        strokeStyle="2" autoSized="false" controlCount="0"
        arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e871c0a8016d2f1d97535a7ece14</URIString>
        <point1 x="1008.7983" y="128.0"/>
        <point2 x="1152.1868" y="241.86163"/>
        <ID1 xsi:type="node">11</ID1>
        <ID2 xsi:type="node">58</ID2>
    </child>
    <child ID="62" layerID="1" created="1554312078054" x="1000.0"
        y="272.14035" width="121.52515" height="1.2270508"
        strokeWidth="1.0" strokeStyle="2" autoSized="false"
        controlCount="0" arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <URIString>http://vue.tufts.edu/rdf/resource/e438e872c0a8016d2f1d97535fc548af</URIString>
        <point1 x="1000.5" y="272.64035"/>
        <point2 x="1121.0251" y="272.8674"/>
        <ID1 xsi:type="node">21</ID1>
        <ID2 xsi:type="node">58</ID2>
    </child>
    <layer ID="1" label="Layer 1" created="1554310875702" x="0.0"
        y="0.0" width="1.4E-45" height="1.4E-45" strokeWidth="0.0" autoSized="false">
        <URIString>http://vue.tufts.edu/rdf/resource/e438e872c0a8016d2f1d9753ef5a6ece</URIString>
    </layer>
    <userZoom>0.728411808901734</userZoom>
    <userOrigin x="-134.0" y="-133.5"/>
    <presentationBackground>#202020</presentationBackground>
    <PathwayList currentPathway="0" revealerIndex="-1">
        <pathway ID="0" label="Untitled Pathway" created="1554310875701"
            x="0.0" y="0.0" width="1.4E-45" height="1.4E-45"
            strokeWidth="0.0" autoSized="false" currentIndex="0" open="true">
            <strokeColor>#B3CC33CC</strokeColor>
            <textColor>#000000</textColor>
            <font>SansSerif-plain-14</font>
            <URIString>http://vue.tufts.edu/rdf/resource/e438e873c0a8016d2f1d9753d141e3e7</URIString>
            <masterSlide ID="2" created="1554310875706" x="0.0" y="0.0"
                width="800.0" height="600.0" locked="true"
                strokeWidth="0.0" autoSized="false">
                <fillColor>#000000</fillColor>
                <strokeColor>#404040</strokeColor>
                <textColor>#000000</textColor>
                <font>SansSerif-plain-14</font>
                <URIString>http://vue.tufts.edu/rdf/resource/e438e873c0a8016d2f1d9753fc24af0a</URIString>
                <titleStyle ID="3" label="Header"
                    created="1554310875713" x="341.0" y="175.0"
                    width="118.0" height="50.0" strokeWidth="0.0"
                    autoSized="true" isStyle="true" xsi:type="node">
                    <strokeColor>#404040</strokeColor>
                    <textColor>#FFFFFF</textColor>
                    <font>Gill Sans-plain-36</font>
                    <URIString>http://vue.tufts.edu/rdf/resource/e438e874c0a8016d2f1d9753c934fc2b</URIString>
                    <shape xsi:type="rectangle"/>
                </titleStyle>
                <textStyle ID="4" label="Slide Text"
                    created="1554310875714" x="349.5" y="282.5"
                    width="101.0" height="35.0" strokeWidth="0.0"
                    autoSized="true" isStyle="true" xsi:type="node">
                    <strokeColor>#404040</strokeColor>
                    <textColor>#FFFFFF</textColor>
                    <font>Gill Sans-plain-22</font>
                    <URIString>http://vue.tufts.edu/rdf/resource/e438e874c0a8016d2f1d97539ac22d1a</URIString>
                    <shape xsi:type="rectangle"/>
                </textStyle>
                <linkStyle ID="5" label="Links" created="1554310875715"
                    x="375.5" y="385.0" width="49.0" height="30.0"
                    strokeWidth="0.0" autoSized="true" isStyle="true" xsi:type="node">
                    <strokeColor>#404040</strokeColor>
                    <textColor>#B3BFE3</textColor>
                    <font>Gill Sans-plain-18</font>
                    <URIString>http://vue.tufts.edu/rdf/resource/e438e875c0a8016d2f1d9753da21c68b</URIString>
                    <shape xsi:type="rectangle"/>
                </linkStyle>
            </masterSlide>
        </pathway>
    </PathwayList>
    <date>2019-04-03</date>
    <modelVersion>6</modelVersion>
    <saveLocation>/Users/jasonbecker/PycharmProjects/dfspy/figures</saveLocation>
    <saveFile>/Users/jasonbecker/PycharmProjects/dfspy/figures/ML flowchart.vue</saveFile>
</LW-MAP>
