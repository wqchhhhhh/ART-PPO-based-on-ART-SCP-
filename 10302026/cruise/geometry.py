"""Spherical geometry; all function arguments use radians and metres."""

import numpy as np


def distance(lon, lat, lon0, lat0, radius=6371000.):
    hav = np.sin((lat-lat0)/2)**2 + np.cos(lat)*np.cos(lat0)*np.sin((lon-lon0)/2)**2
    return 2*radius*np.arcsin(np.sqrt(np.clip(hav, 0., 1.)))


def bearing(lon0, lat0, lon1, lat1):
    return np.arctan2(np.sin(lon1-lon0)*np.cos(lat1),
                      np.cos(lat0)*np.sin(lat1)-np.sin(lat0)*np.cos(lat1)*np.cos(lon1-lon0))


def great_circle(lon0, lat0, lon1, lat1, progress):
    def cart(lon, lat):
        return np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    a, b = cart(lon0, lat0), cart(lon1, lat1)
    angle = np.arccos(np.clip(a@b, -1., 1.))
    s = np.asarray(progress)
    p = (np.sin((1-s)*angle)[:, None]*a + np.sin(s*angle)[:, None]*b)/np.sin(angle)
    return np.arctan2(p[:, 1], p[:, 0]), np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1]))


def clearances(states, mission, radius):
    x = np.asarray(states)
    return np.stack([distance(x[..., 1], x[..., 2], np.radians(z.longitude_deg),
                              np.radians(z.latitude_deg), radius)-z.radius_km*1000
                     for z in mission.zones], axis=-1) if mission.zones else np.empty((*x.shape[:-1], 0))
