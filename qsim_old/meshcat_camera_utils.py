import meshcat


def SetOrthographicCameraYZ(vis: meshcat.Visualizer) -> None:
    # use orthographic camera, show YZ plane.
    camera = meshcat.geometry.OrthographicCamera(
        left=-0.5, right=2, bottom=-1, top=4, near=-1000, far=1000
    )
    vis["/Cameras/default/rotated"].set_object(camera)
    vis["/Cameras/default/rotated/<object>"].set_property("position", [0, 0, 0])
    vis["/Cameras/default"].set_transform(
        meshcat.transformations.translation_matrix([1, 0, 0])
    )


def SetOrthographicCameraXY(vis: meshcat.Visualizer) -> None:
    # use orthographic camera, show xy plane.
    camera = meshcat.geometry.OrthographicCamera(
        left=-0.5, right=0.5, bottom=-0.5, top=0.5, near=-1000, far=1000
    )
    vis["/Cameras/default/rotated"].set_object(camera)
    vis["/Cameras/default"].set_transform(
        meshcat.transformations.translation_matrix([0, 0, 1])
    )
