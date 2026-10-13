import ast, io, sys, time, gc, json, platform, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'files'))
source=root/'files/lfi_viewer_final.py'
tree=ast.parse(source.read_text())
# Execute the real application helpers without running its Streamlit UI.
nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.Try))]
namespace={}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
data=namespace['smart_load'](str(root/'Realdata/mushroom.hdf5'))
for channel in ('amplitude','phase'):
 raw=data.get_slice(data.z_indices[0])[channel]
 recovered=np.loadtxt(io.StringIO(namespace['export_csv'](raw)),delimiter=',')
 assert raw.shape==recovered.shape
 assert np.allclose(raw,recovered)
 print('CSV PASS',channel,raw.shape)
X,Y,Z,values,step=namespace['get_downsampled_volume'].__wrapped__(data)
trace=namespace['create_3d_trace'](X,Y,Z,values,'Viridis','isosurface')
assert trace.type=='isosurface' and trace.surface.count==3
assert trace.isomin==np.percentile(values,60) and trace.isomax==np.percentile(values,95)
assert not trace.caps.x.show and not trace.caps.y.show and not trace.caps.z.show
print('ISOSURFACE PASS (single plane mushroom; object validation only)')
raw=data.amplitude[0]; norm=(raw-raw.min())/(raw.max()-raw.min()+1e-10)
for cmap in ('Viridis','Cool','ice','missing'):
 png=namespace['publication_png'](norm,cmap,10)
 dpi=Image.open(io.BytesIO(png)).info['dpi']
 assert np.allclose(dpi,(300,300),atol=.01)
print('PUBLICATION DPI PASS',dpi)
from streamlit.testing.v1 import AppTest
app=AppTest.from_file(str(source),default_timeout=30).run()
app.checkbox[0].check().run()
app.text_input[0].set_value(str(root/'Realdata/mushroom.hdf5')).run()
next(b for b in app.button if b.label=='📥 Load from Path').click().run()
assert not app.exception, app.exception
assert any(n.label=='Pixel X' for n in app.number_input)
next(n for n in app.number_input if n.label=='Pixel size (μm)').set_value(10.).run()
assert not app.exception,app.exception
print('APPTEST PASS')
times=[]
for i in range(5):
 gc.collect()
 start=time.perf_counter()
 loaded=namespace['smart_load'](str(root/'Realdata/chip_borad1.hdf5'))
 elapsed=time.perf_counter()-start
 assert loaded.amplitude.shape==(70,1400,400)
 times.append(elapsed)
 del loaded
 print('TIMING',i+1,elapsed)
print('RESULT',json.dumps({'seconds':times,'mean':float(np.mean(times)),'population_std':float(np.std(times)),'sample_std':float(np.std(times,ddof=1)),'python':sys.version,'platform':platform.platform()}))
print(subprocess.check_output(['system_profiler','SPHardwareDataType'],text=True).split('Serial Number')[0])
