import { Frustum, Matrix4, Vector4 } from 'three/webgpu';
import { uniform } from 'three/tsl';

export function createFrustumPlaneState() {
  return {
    frustum: new Frustum(),
    projectionView: new Matrix4(),
    uniforms: Array.from({ length: 6 }, () => uniform(new Vector4())),
  };
}

export function updateFrustumPlaneState(state, camera, renderer) {
  camera.updateMatrixWorld();
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  state.projectionView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  state.frustum.setFromProjectionMatrix(
    state.projectionView,
    renderer.coordinateSystem,
    camera.reversedDepth,
  );
  for (let index = 0; index < state.uniforms.length; index += 1) {
    const plane = state.frustum.planes[index];
    state.uniforms[index].value.set(
      plane.normal.x,
      plane.normal.y,
      plane.normal.z,
      plane.constant,
    );
  }
}
