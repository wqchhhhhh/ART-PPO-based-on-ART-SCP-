"""
Constraints module for Air-breathing Hypersonic Vehicle (AHV) Trajectory Optimization.
Defines boundary conditions, path constraints, and no-fly zone exclusion conditions.
Supports parameterized no-fly zone sets for multi-scenario testing.
"""

import casadi as ca
import numpy as np

import config as cfg


class Constraints:
    """
    Evaluates path and no-fly zone constraint expressions in CasADi.
    """

    @staticmethod
    def prepare_nfzs(nfz_list):
        """
        Ensure all NFZs have rad angles and cosine thresholds computed.
        """
        prepared = []
        for nfz in nfz_list:
            item = dict(nfz)
            item["lon_rad"] = np.radians(item["lon_deg"])
            item["lat_rad"] = np.radians(item["lat_deg"])
            item["angular_radius_rad"] = (item["radius_km"] * 1000.0) / cfg.R_E
            item["cos_radius"] = np.cos(item["angular_radius_rad"])
            prepared.append(item)
        return prepared

    @staticmethod
    def no_fly_zone_cosines(theta, phi, nfzs=None):
        """
        Compute spherical distance cosine for each no-fly zone:
        cos(d / R_E) = sin(phi) * sin(phi_k) + cos(phi) * cos(phi_k) * cos(theta - theta_k)
        
        Condition for point being OUTSIDE NFZ:
            cos(d / R_E) <= cos(R_k / R_E)
            or: cos(d / R_E) - cos(R_k / R_E) <= 0
        """
        if nfzs is None:
            nfzs = cfg.NO_FLY_ZONES

        g_nfz = []
        for nfz in nfzs:
            lat_rad = nfz.get("lat_rad", np.radians(nfz["lat_deg"]))
            lon_rad = nfz.get("lon_rad", np.radians(nfz["lon_deg"]))
            cos_rad = nfz.get("cos_radius", np.cos((nfz["radius_km"] * 1000.0) / cfg.R_E))

            cos_d = (
                ca.sin(phi) * np.sin(lat_rad)
                + ca.cos(phi) * np.cos(lat_rad) * ca.cos(theta - lon_rad)
            )
            g_nfz.append(cos_d - cos_rad)
        return ca.vertcat(*g_nfz)

    @staticmethod
    def no_fly_zone_distances_km(lons_deg, lats_deg, nfzs=None):
        """
        Compute true great-circle distance [km] from an array of coordinates to all NFZs.
        Used for verification and visualization.
        """
        if nfzs is None:
            nfzs = cfg.NO_FLY_ZONES

        distances = {}
        for nfz in nfzs:
            lat_rad = nfz.get("lat_rad", np.radians(nfz["lat_deg"]))
            lon_rad = nfz.get("lon_rad", np.radians(nfz["lon_deg"]))
            r_km = nfz["radius_km"]

            phi1 = np.radians(lats_deg)
            phi2 = lat_rad
            dphi = phi2 - phi1
            dlam = lon_rad - np.radians(lons_deg)
            a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
            d_km = 2.0 * (cfg.R_E / 1000.0) * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
            distances[nfz["name"]] = {
                "dist_km": d_km,
                "radius_km": r_km,
                "clearance_km": d_km - r_km
            }
        return distances

    @staticmethod
    def scaled_no_fly_zone_residuals(theta, phi, nfzs, buffer_m=0.0):
        """Smooth exclusions scaled to kilometres near each NFZ boundary."""
        residuals = []
        for nfz in nfzs:
            radius_m = nfz["radius_km"] * 1000.0 + buffer_m
            cos_d = (ca.sin(phi) * np.sin(nfz["lat_rad"])
                     + ca.cos(phi) * np.cos(nfz["lat_rad"])
                     * ca.cos(theta - nfz["lon_rad"]))
            residuals.append((cos_d - np.cos(radius_m / cfg.R_E))
                             * cfg.R_E ** 2 / (radius_m * 1000.0))
        return ca.vertcat(*residuals)

    @staticmethod
    def no_fly_zone_boundary(nfz,count=181):
        """Geodesic circle in longitude/latitude degrees for plotting."""
        lon,lat = np.radians([nfz["lon_deg"],nfz["lat_deg"]])
        radius = nfz["radius_km"]*1000./cfg.R_E
        bearing = np.linspace(0.,2*np.pi,count)
        lat2 = np.arcsin(np.sin(lat)*np.cos(radius)+np.cos(lat)*np.sin(radius)*np.cos(bearing))
        lon2 = lon+np.arctan2(np.sin(bearing)*np.sin(radius)*np.cos(lat),
                             np.cos(radius)-np.sin(lat)*np.sin(lat2))
        return np.degrees(lon2),np.degrees(lat2)
